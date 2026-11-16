import torch
import triton
import triton.language as tl


@triton.jit
def vector_add(
    a,
    b,
    c,
    n: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n

    x = tl.load(a + offsets, mask=mask, other=0)
    y = tl.load(b + offsets, mask=mask, other=0)
    tl.store(c + offsets, x + y, mask=mask)



def test():
    n = 4096
    a = torch.rand(n, device="cuda")
    b = torch.rand(n, device="cuda")
    c = torch.empty_like(a)

    vector_add[(triton.cdiv(n, 256),)](a, b, c, n, 256) # pyright: ignore[reportArgumentType]

    torch.testing.assert_close(c, a + b)
    print("Triton vector_add passed!")


if __name__ == "__main__":
    test()
