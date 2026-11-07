"""Decode lossless model-input tensors emitted by ``ReplayTensorCodec``.

The transport contains observations, not recoverable game snapshots. Decoding
uses only the standard library; tensor data stays in immutable byte-backed
memoryviews instead of expanding into Python numbers. NumPy is imported only
by ``as_numpy`` or ``collate_observations``. Schema identifiers are deliberately
opaque here: a trainer should pass its expected full schema identifier to
``decode_observation`` so a changed catalog cannot silently change its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any, Iterable
import zlib


_MAGIC = 0x31454D41
_ELEMENT_BYTES = {"float32": 4, "int32": 4, "uint8": 1, "bool": 1}
_NUMPY_TYPES = {"float32": "<f4", "int32": "<i4", "uint8": "u1", "bool": "?"}
_INT32 = struct.Struct("<i")
_UINT32 = struct.Struct("<I")


class TensorDecodeError(ValueError):
    """The binary observation is malformed, incompatible, or over its limits."""


@dataclass(frozen=True)
class DecodeLimits:
    max_decoded_bytes: int = 64 * 1024 * 1024
    max_encoded_bytes: int = 65 * 1024 * 1024
    max_tensors: int = 256
    max_rank: int = 8
    max_string_bytes: int = 4096

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class TensorBuffer:
    name: str
    shape: tuple[int, ...]
    dtype: str
    data: memoryview

    @property
    def nbytes(self) -> int:
        return self.data.nbytes

    @property
    def element_count(self) -> int:
        return self.nbytes // _ELEMENT_BYTES[self.dtype]

    def as_numpy(self, *, copy: bool = False) -> Any:
        """Return a correctly shaped little-endian array, read-only by default.

        ``copy=False`` shares the decoded byte storage. ``copy=True`` produces a
        writable independent array suitable for a caller that modifies inputs.
        Batching also copies once into a writable contiguous batch array.
        """
        import numpy as np

        result = np.frombuffer(self.data, dtype=_NUMPY_TYPES[self.dtype]).reshape(self.shape)
        return result.copy() if copy else result


@dataclass(frozen=True)
class EncodedObservation:
    schema: str
    tensors: dict[str, TensorBuffer]

    @property
    def tensor_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors.values())

    def as_numpy(self, *, copy: bool = False) -> dict[str, Any]:
        return {name: tensor.as_numpy(copy=copy) for name, tensor in self.tensors.items()}


@dataclass(frozen=True)
class TensorBatch:
    schema: str
    tensors: dict[str, Any]
    shapes: dict[str, Any]


class _Reader:
    def __init__(self, data: memoryview, limits: DecodeLimits):
        self.data = data
        self.offset = 0
        self.limits = limits

    def take(self, length: int) -> memoryview:
        if length < 0 or length > len(self.data) - self.offset:
            raise TensorDecodeError("Truncated tensor record")
        result = self.data[self.offset:self.offset + length]
        self.offset += length
        return result

    def integer(self) -> int:
        return _INT32.unpack(self.take(4))[0]

    def string(self) -> str:
        # BinaryWriter writes the UTF-8 *byte* length as a .NET 7-bit int.
        length = 0
        for shift in range(0, 35, 7):
            value = self.take(1)[0]
            if shift == 28 and value > 7:
                raise TensorDecodeError("Invalid 7-bit string length")
            length |= (value & 0x7F) << shift
            if value < 0x80:
                break
        else:
            raise TensorDecodeError("Invalid 7-bit string length")
        if length > self.limits.max_string_bytes:
            raise TensorDecodeError("Tensor string exceeds its configured limit")
        try:
            return self.take(length).tobytes().decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise TensorDecodeError("Invalid UTF-8 tensor metadata") from error


def decode_observation(
    payload: bytes | bytearray | memoryview,
    *,
    expected_schema: str | None = None,
    limits: DecodeLimits | None = None,
) -> EncodedObservation:
    """Decode a raw (0) or zlib (2) .NET tensor record, validating all lengths.

    Legacy envelope 1 uses Brotli and is rejected explicitly; current .NET
    writers emit zlib when compression is enabled. The encoded and decoded
    byte limits also bound compressed records before tensor metadata is read.
    Immutable ``bytes`` input is shared for raw records; mutable input is
    frozen so subsequent caller writes cannot change an observation.
    """
    limits = limits or DecodeLimits()
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes, bytearray, or memoryview")
    if (payload.nbytes if isinstance(payload, memoryview) else len(payload)) > limits.max_encoded_bytes:
        raise TensorDecodeError("Encoded observation exceeds its configured limit")
    owned = payload if isinstance(payload, bytes) else bytes(payload)
    if not owned:
        raise TensorDecodeError("Empty tensor record")
    envelope, contents = owned[0], memoryview(owned)[1:]
    if envelope == 0:
        if len(contents) > limits.max_decoded_bytes:
            raise TensorDecodeError("Decoded observation exceeds its configured limit")
        raw = contents
    elif envelope == 2:
        decoder = zlib.decompressobj()
        try:
            decoded = decoder.decompress(contents, limits.max_decoded_bytes + 1)
        except zlib.error as error:
            raise TensorDecodeError("Invalid zlib tensor record") from error
        if len(decoded) > limits.max_decoded_bytes or decoder.unconsumed_tail:
            raise TensorDecodeError("Decoded observation exceeds its configured limit")
        if not decoder.eof:
            raise TensorDecodeError("Truncated zlib tensor record")
        if decoder.unused_data:
            raise TensorDecodeError("Trailing data after compressed tensor record")
        raw = memoryview(decoded)
    elif envelope == 1:
        raise TensorDecodeError("Legacy Brotli tensor records are unsupported; export zlib or raw from .NET")
    else:
        raise TensorDecodeError("Unknown tensor compression envelope")

    reader = _Reader(raw, limits)
    if _UINT32.unpack(reader.take(4))[0] != _MAGIC:
        raise TensorDecodeError("Unknown tensor transport magic/version")
    schema = reader.string()
    if not schema:
        raise TensorDecodeError("Empty tensor schema identifier")
    if expected_schema is not None and schema != expected_schema:
        raise TensorDecodeError(f"Tensor schema mismatch: expected {expected_schema!r}, received {schema!r}")
    count = reader.integer()
    if count < 0 or count > limits.max_tensors:
        raise TensorDecodeError("Invalid tensor registry size")
    tensors: dict[str, TensorBuffer] = {}
    for _ in range(count):
        name = reader.string()
        if not name or name in tensors:
            raise TensorDecodeError("Empty or duplicate tensor name")
        rank = reader.integer()
        if rank < 0 or rank > limits.max_rank:
            raise TensorDecodeError("Invalid tensor rank")
        shape = tuple(reader.integer() for _ in range(rank))
        elements = 1
        for dimension in shape:
            if dimension < 0:
                raise TensorDecodeError("Negative tensor dimension")
            elements *= dimension
        dtype = reader.string()
        if dtype not in _ELEMENT_BYTES:
            raise TensorDecodeError(f"Unsupported tensor dtype: {dtype!r}")
        length = reader.integer()
        if length < 0 or length != elements:
            raise TensorDecodeError("Tensor element count does not match its shape")
        byte_count = length * _ELEMENT_BYTES[dtype]
        if byte_count > limits.max_decoded_bytes:
            raise TensorDecodeError("Tensor exceeds its configured byte limit")
        tensors[name] = TensorBuffer(name, shape, dtype, reader.take(byte_count))
    if reader.offset != len(raw):
        raise TensorDecodeError("Trailing data after tensor registry")
    return EncodedObservation(schema, tensors)


def collate_observations(
    observations: Iterable[EncodedObservation],
    *,
    max_batch_bytes: int = 512 * 1024 * 1024,
) -> TensorBatch:
    """Create writable NumPy batches, zero-padding variable entity dimensions.

    Names, rank, dtype and full schema must agree. ``shapes[name][i]`` records
    every unpadded input shape; the trainer can derive masks for variable-size
    axes. Explicit masks already present in the observation remain ordinary
    tensors. This function does not invent game semantics or policy masks.
    """
    import numpy as np

    records = tuple(observations)
    if not records:
        raise ValueError("Cannot collate an empty observation batch")
    if not isinstance(max_batch_bytes, int) or isinstance(max_batch_bytes, bool) or max_batch_bytes < 1:
        raise ValueError("max_batch_bytes must be a positive integer")
    first = records[0]
    names = tuple(first.tensors)
    for record in records:
        if record.schema != first.schema:
            raise ValueError("Cannot batch observations from different schemas")
        if record.tensors.keys() != first.tensors.keys():
            raise ValueError("Cannot batch observations with different tensor registries")
    plan: list[tuple[str, tuple[int, ...], str]] = []
    total_bytes = 0
    for name in names:
        template = first.tensors[name]
        rank = len(template.shape)
        for record in records:
            tensor = record.tensors[name]
            if tensor.dtype != template.dtype or len(tensor.shape) != rank:
                raise ValueError(f"Inconsistent dtype or rank for tensor {name!r}")
        shape = tuple(max(record.tensors[name].shape[axis] for record in records) for axis in range(rank))
        elements = len(records)
        for dimension in shape:
            elements *= dimension
        total_bytes += elements * _ELEMENT_BYTES[template.dtype] + len(records) * rank * 4
        if total_bytes > max_batch_bytes:
            raise ValueError("Padded batch exceeds its configured byte limit")
        plan.append((name, shape, template.dtype))
    tensors: dict[str, Any] = {}
    shapes: dict[str, Any] = {}
    for name, shape, dtype in plan:
        batch = np.zeros((len(records), *shape), dtype=_NUMPY_TYPES[dtype])
        for index, record in enumerate(records):
            tensor = record.tensors[name]
            batch[(index, *(slice(0, dimension) for dimension in tensor.shape))] = tensor.as_numpy()
        tensors[name] = batch
        shapes[name] = np.asarray([record.tensors[name].shape for record in records], dtype=np.int32)
    return TensorBatch(first.schema, tensors, shapes)
