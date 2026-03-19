# Engineering Notes on INT4 Packing and CUTLASS W4A8 MoE Inputs

This note consolidates the practical conclusions from debugging INT4 packing and the `cutlass_w4a8_moe` path in SGLang. It is intended for engineers who need to answer questions such as:

- What exactly is a nibble?
- How do logical INT4 values relate to packed `int8` and packed `int32` tensors?
- When is `int32.view(torch.int8)` correct, and when is it wrong?
- What are the exact weight and scale layout requirements for `test_cutlass_w4a8_moe.py` and `cutlass_w4a8_moe`?
- Does the CUTLASS W4A8 kernel expect a scale or its reciprocal?

The focus here is on the engineering contract between the Python-side tensor preparation code and the CUDA/CUTLASS kernel.

## Terminology

### Nibble

A **nibble** is 4 bits.

- `1 byte = 8 bits = 2 nibbles`
- `int4` occupies 1 nibble
- `int8` occupies 2 nibbles
- `int32` occupies 8 nibbles

This is why INT4 quantization always involves packing: the storage primitive is usually at least 1 byte, while each logical value only needs 4 bits.

## Three Different Representations of "INT4 Weights"

When discussing "INT4 weights", it is critical to distinguish three different tensor representations.

### 1. Logical INT4 tensor stored in `torch.int8`

PyTorch does not have a native `torch.int4` dtype, so the common representation is:

- dtype: `torch.int8`
- valid value range: `[-8, 7]`

Example:

```python
ref_weight = torch.randint(-8, 8, (E, N, K), dtype=torch.int8)
```

This is **not packed yet**. Each tensor element is one logical INT4 value carried in an `int8` container.

### 2. Packed INT4 tensor stored in `torch.int8`

Two INT4 values are packed into one `int8`:

```text
packed_byte = [high_nibble : low_nibble]
```

In SGLang's CUTLASS W4A8 tests, the convention is:

- even-indexed INT4 value goes to the low nibble
- odd-indexed INT4 value goes to the high nibble

The reference implementation is:

```python
low_nibbles = input_tensor_int8[..., 0::2]
high_nibbles = input_tensor_int8[..., 1::2]
packed_tensor = (high_nibbles << 4) | (low_nibbles & 0x0F)
```

So each output byte stores:

```text
packed[..., i] = [input[..., 2*i + 1] : input[..., 2*i]]
```

### 3. Packed INT4 tensor stored in `torch.int32`

Eight INT4 values can be packed into one `int32`. A natural packing order is:

```text
bits[ 3: 0] = v0
bits[ 7: 4] = v1
bits[11: 8] = v2
bits[15:12] = v3
bits[19:16] = v4
bits[23:20] = v5
bits[27:24] = v6
bits[31:28] = v7
```

Under this natural order, the same 32-bit payload is also equivalent to four packed bytes:

```text
byte0 = [v1:v0]
byte1 = [v3:v2]
byte2 = [v5:v4]
byte3 = [v7:v6]
```

This equivalence is exactly why `int32.view(torch.int8)` is sometimes valid and sometimes not.

## How `pack_to_int32()` Works

Consider a function of the following form:

```python
def pack_to_int32(value, num_bits, packed_dim=1):
    ...
```

Its goal is to pack low-bit values into 32-bit containers.

### Step 1: Convert signed values into an unsigned code space

A common implementation uses:

```python
offset = 1 << (num_bits - 1)
value = (value + offset).to(torch.uint8)
```

For `num_bits = 4`, this means:

```text
-8 -> 0
-7 -> 1
...
-1 -> 7
 0 -> 8
...
 7 -> 15
```

This is an **offset encoding**. It is not the same thing as directly preserving the signed INT4 two's-complement bit pattern.

This distinction matters because a downstream kernel may expect either:

- packed offset-coded values, or
- packed two's-complement INT4 bit patterns.

If the packer and the kernel disagree on this encoding contract, the result will be wrong even if all tensor shapes and dtypes look valid.

### Step 2: Compute the packing factor

```python
pack_factor = 32 // num_bits
```

Examples:

- INT4: `pack_factor = 8`
- INT8: `pack_factor = 4`

### Step 3: Pad the packed dimension if needed

Packing requires groups of exactly `pack_factor` values. If the source dimension is not divisible by that factor, zero-padding is added before packing and removed during unpacking.

### Step 4: Shift each code into its bit slot and sum

For INT4, the shift pattern is:

```text
[0, 4, 8, 12, 16, 20, 24, 28]
```

The packed 32-bit value is then:

```text
(v0 << 0) + (v1 << 4) + (v2 << 8) + ... + (v7 << 28)
```

Because each value occupies a disjoint 4-bit interval, this sum is equivalent to a bitwise OR.

### Resulting layout

The implementation above produces a **natural nibble order**:

```text
lowest nibble  <- first logical value
highest nibble <- last logical value
```

## How `unpack_from_int32()` Works

An unpacking function for this scheme usually:

1. extracts each `num_bits`-wide field via shift and mask,
2. restores the original tensor ordering,
3. removes pack-time padding,
4. subtracts the offset to recover the signed logical value.

For INT4, the extraction pattern is:

```python
mask = 0x0F
field_i = (value >> (4 * i)) & mask
```

This pulls out the eight nibbles from an `int32`.

### What does the unpack function return?

If the input is one packed `int32` containing eight INT4 values, the unpack result is:

- eight tensor elements,
- each stored as `torch.int8`,
- each logically representing one INT4 value in `[-8, 7]`.

It does **not** return four packed bytes. It returns eight unpacked logical values.

### Is unpack complete with respect to the pack function?

Yes, provided the following conditions hold:

1. `num_bits` matches between pack and unpack
2. `packed_dim` matches
3. the original shape is provided so padding can be removed
4. the original values were within the valid representable range
5. the input really came from the matching pack function

In that setting, unpack is the inverse of pack.

## `view(torch.int8)` Is Not the Same as Unpacking

This is one of the most common sources of confusion.

### `int32.view(torch.int8)`

`view(torch.int8)` only changes how the same memory is interpreted. It does **not**:

- unpack INT4 values,
- repack values,
- reorder nibbles,
- or correct signedness.

If a tensor contains one `int32`, then `view(torch.int8)` exposes the same payload as four bytes.

### Real unpacking

A real unpack function performs:

- shifts,
- masks,
- possible reordering,
- and signed-value restoration.

Therefore:

- `view(torch.int8)` gives **four packed bytes**
- unpacking gives **eight logical INT4 values**

These operations serve different purposes.

## When Is `int32.view(torch.int8)` Correct?

`int32.view(torch.int8)` is correct only if the existing `int32` bit layout already matches the packed-byte convention required by the consumer.

### Case A: natural nibble order

If the 32-bit integer stores:

```text
[v7][v6][v5][v4][v3][v2][v1][v0]
```

with natural nibble order, then under little-endian byte ordering the four bytes are:

```text
[v1:v0], [v3:v2], [v5:v4], [v7:v6]
```

That is exactly the packed-`int8` convention used in the CUTLASS W4A8 tests.

In this case, `view(torch.int8)` can be valid.

### Case B: reordered nibble layout

Some packers do not use natural order. For example, a reorder map like:

```text
[0, 2, 4, 6, 1, 3, 5, 7]
```

produces this nibble sequence inside the `int32`:

```text
nibble0 = v0
nibble1 = v2
nibble2 = v4
nibble3 = v6
nibble4 = v1
nibble5 = v3
nibble6 = v5
nibble7 = v7
```

Now the four bytes become:

```text
[v2:v0], [v6:v4], [v3:v1], [v7:v5]
```

This no longer matches the expected packed-byte sequence:

```text
[v1:v0], [v3:v2], [v5:v4], [v7:v6]
```

In this case, `view(torch.int8)` is wrong even though the dtype and the byte count look correct.

## CUTLASS W4A8 MoE: Weight Requirements

The CUTLASS W4A8 MoE path uses two different weight tensors:

- `w1_q` for the fused gate/up projection
- `w2_q` for the down projection

The high-level wrapper documents them as:

- `w1_q`: `[num_experts, 2N, K/2]`
- `w2_q`: `[num_experts, K, N/2]`

where each `int8` stores two packed INT4 values.

### Reference weight layout before packing

The tests create:

```python
ref_weight_1 = torch.randint(-8, 8, (E, 2N, K), dtype=torch.int8)
ref_weight_2 = torch.randint(-8, 8, (E, K, N), dtype=torch.int8)
```

These tensors use `torch.int8` only as a container. Their logical value range is INT4.

### Weight packing rule used by the test

The helper:

```python
pack_int4_values_to_int8()
```

packs along the **last dimension**.

Therefore:

- `ref_weight_1 [E, 2N, K] -> w1_q [E, 2N, K/2]`
- `ref_weight_2 [E, K, N] -> w2_q [E, K, N/2]`

This means:

- `w1_q` is packed along the `K` dimension
- `w2_q` is packed along the `N` dimension

That distinction is essential. One cannot use a single generic "pack to int32 then view as int8" rule without first verifying which logical dimension is being packed.

### Why the documentation says "transposed and packed"

For the first GEMM:

```text
A [M, K] @ W1^T [K, 2N] -> [M, 2N]
```

The stored reference weight is `[2N, K]`, which is the standard `[out_features, in_features]` arrangement for a linear layer. The CUTLASS wrapper treats this as the B operand in a column-major layout, hence the documentation wording.

For the second GEMM:

```text
A2 [M, N] @ W2^T [N, K] -> [M, K]
```

The stored reference weight is `[K, N]`, again in `[out_features, in_features]` form.

The practical takeaway is simple:

- the logical reference weight is always stored as `[out_dim, in_dim]`
- packing happens along the reduction dimension

## CUTLASS W4A8 MoE: Scale Requirements

The scale layout is more subtle than the weight layout because the kernel expects an interleaved packing of scale blocks.

### Group size

The W4A8 path uses a group size of 128 weights per scale value.

### Reference scale for the first GEMM

The test builds:

```python
scale_1: [E, 2N, K / 128]
```

Interpretation:

```text
scale_1[e, out_c, g]
```

applies to:

```text
ref_weight_1[e, out_c, g*128 : (g+1)*128]
```

### Reference scale for the second GEMM

The test builds:

```python
scale_2: [E, K, N / 128]
```

Interpretation:

```text
scale_2[e, out_c, g]
```

applies to:

```text
ref_weight_2[e, out_c, g*128 : (g+1)*128]
```

### How the reference implementation uses the scale

The reference path repeats each scale value over its 128-element group:

```python
ref_w_scale_repeat = ref_weight_scale[e].repeat_interleave(128, dim=1)
ref_weight_dequant = ref_weight[e].float() * ref_w_scale_repeat.float()
```

This means the scale is a direct multiplicative dequantization factor.

### Interleaving required by the kernel

The kernel does not consume the scale in the simple logical form above. The test performs:

```python
scale_interleaved = ref_scale.reshape(E, out_dim, groups / alignment, alignment)
scale_interleaved = scale_interleaved.permute(0, 2, 1, 3)
scale_interleaved = scale_interleaved.reshape(E, groups / alignment, out_dim * alignment)
```

This reorganizes the scale layout for efficient kernel access.

#### Typical shapes

For the first GEMM:

- reference scale: `[E, 2N, K/128]`
- interleaved scale: `[E, K/512, 2N * 4]` when alignment is 4

For the second GEMM:

- reference scale: `[E, K, N/128]`
- interleaved scale: `[E, N/512, K * 4]` when alignment is 4

If the reduction dimension is not compatible with the 4-way packing, the code falls back to `alignment = 1`.

### Important: physical shape and logical stride are not the same thing

The physical tensor shape of `w_scale` may end with `2N * 4` or `K * 4`, but the stride tensor passed to the kernel still corresponds to logical output-channel indexing rather than that raw physical last dimension.

This is because the kernel treats each scale access unit as a packed structure rather than as an arbitrary scalar BF16 array.

## Does the CUTLASS W4A8 Kernel Expect Scale or Reciprocal Scale?

This is a recurring source of confusion because some variables in the codebase are named `*_scale_inv`.

### Conclusion

For the CUTLASS W4A8 MoE path, the kernel expects a **direct multiplicative scale**, not its reciprocal.

In other words:

```text
weight_real = weight_quant * scale
```

not:

```text
weight_real = weight_quant / scale
```

### Evidence

1. The Python reference implementation multiplies the reference INT4 weights by the repeated scale values.
2. The W4A8 test passes the scale tensors directly after interleaving. There is no `reciprocal()` call in that path.
3. The CUTLASS wrapper documents the operation as:

   ```text
   D = A * (B * scales)
   ```

4. The `w4afp8.py` path uses variable names like `w13_weight_scale_inv`, but before calling the CUTLASS MoE wrapper it only:

   - converts dtype,
   - interleaves the scale,
   - and forwards it.

   It does not invert the scale numerically.

The most reasonable interpretation is that `*_scale_inv` is a reused naming convention in the broader codebase, not the mathematical contract of this specific CUTLASS path.

## Practical Validation Workflow

When introducing a new packer or checkpoint conversion path, validate in two stages.

### Stage 1: Byte-level validation

Compare the packed bytes produced by:

1. the known-good reference packer:

   ```python
   packed_int8_ref = (high << 4) | (low & 0x0F)
   ```

2. your proposed `int32` path:

   ```python
   packed_int8_candidate = packed_int32.view(torch.int8)
   ```

Then compare raw bytes:

```python
torch.equal(
    packed_int8_ref.view(torch.uint8),
    packed_int8_candidate.view(torch.uint8),
)
```

If this check fails, the `int32` path is not layout-compatible with the kernel input format.

### Stage 2: End-to-end kernel validation

Run the same weights and scales through:

- the reference path from the test, and
- the CUTLASS kernel path,

then compare outputs with `torch.testing.assert_close`.

Only if both checks pass should a new packing path be considered compatible.

## Common Failure Modes

### 1. Treating `view(torch.int8)` as automatic repacking

It is not. It only reinterprets bytes.

### 2. Forgetting that different packers may use different nibble order

Natural order and reordered layouts are not interchangeable.

### 3. Forgetting that different packers may use different encoding

Offset-coded INT4 and two's-complement INT4 are different formats.

### 4. Packing the wrong dimension

For CUTLASS W4A8 MoE:

- `w1_q` packs along `K`
- `w2_q` packs along `N`

### 5. Assuming scale tensors can be passed in their logical shape

The kernel expects an interleaved scale layout, not the raw logical `[out_dim, groups]` layout used by the reference implementation.

### 6. Assuming `*_scale_inv` means the CUTLASS path wants reciprocal scale

It does not in this case.

## Checklist for Engineers

Before passing INT4 weights to `cutlass_w4a8_moe`, verify all of the following:

- The logical source values are in the INT4 range `[-8, 7]`
- The final weight tensors are `torch.int8`
- Each output byte stores exactly two INT4 values in the expected nibble order
- `w1_q` is packed along `K`
- `w2_q` is packed along `N`
- The scale tensors are BF16 on the CUDA device
- The scale tensors are interleaved into the kernel's expected physical layout
- The scale values are direct dequantization multipliers, not reciprocals
- Any `int32.view(torch.int8)` shortcut has been byte-validated against the reference packer

## Summary

The engineering contract for the CUTLASS W4A8 MoE path is stricter than "use INT4 weights".

Correct execution depends on all of the following matching the kernel contract:

- value encoding,
- nibble order,
- packed dimension,
- weight tensor shape,
- scale grouping,
- scale interleaving,
- and the mathematical meaning of the scale tensor.

If any one of these is wrong, the program may still run with seemingly valid tensor shapes and dtypes, but the numerical result will be incorrect.
