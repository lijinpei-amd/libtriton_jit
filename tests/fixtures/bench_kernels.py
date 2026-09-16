# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import triton
import triton.language as tl


@triton.jit
def launch_4(p0, p1, p2, p3):
    value = tl.load(p1) + tl.load(p2) + tl.load(p3)
    tl.store(p0, value)


@triton.jit
def launch_11(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10):
    value = (
        tl.load(p1)
        + tl.load(p2)
        + tl.load(p3)
        + tl.load(p4)
        + tl.load(p5)
        + tl.load(p6)
        + tl.load(p7)
        + tl.load(p8)
        + tl.load(p9)
        + tl.load(p10)
    )
    tl.store(p0, value)


@triton.jit
def launch_30(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    p16,
    p17,
    p18,
    p19,
    p20,
    p21,
    p22,
    p23,
    p24,
    p25,
    p26,
    p27,
    p28,
    p29,
):
    value = (
        tl.load(p1)
        + tl.load(p2)
        + tl.load(p3)
        + tl.load(p4)
        + tl.load(p5)
        + tl.load(p6)
        + tl.load(p7)
        + tl.load(p8)
        + tl.load(p9)
        + tl.load(p10)
        + tl.load(p11)
        + tl.load(p12)
        + tl.load(p13)
        + tl.load(p14)
        + tl.load(p15)
        + tl.load(p16)
        + tl.load(p17)
        + tl.load(p18)
        + tl.load(p19)
        + tl.load(p20)
        + tl.load(p21)
        + tl.load(p22)
        + tl.load(p23)
        + tl.load(p24)
        + tl.load(p25)
        + tl.load(p26)
        + tl.load(p27)
        + tl.load(p28)
        + tl.load(p29)
    )
    tl.store(p0, value)


@triton.jit(do_not_specialize=range(30))
def launch_dynamic_nospec_30(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    p16,
    p17,
    p18,
    p19,
    p20,
    p21,
    p22,
    p23,
    p24,
    p25,
    p26,
    p27,
    p28,
    p29,
):
    value = (
        tl.load(p1)
        + tl.load(p2)
        + tl.load(p3)
        + tl.load(p4)
        + tl.load(p5)
        + tl.load(p6)
        + tl.load(p7)
        + tl.load(p8)
        + tl.load(p9)
        + tl.load(p10)
        + tl.load(p11)
        + tl.load(p12)
        + tl.load(p13)
        + tl.load(p14)
        + tl.load(p15)
        + tl.load(p16)
        + tl.load(p17)
        + tl.load(p18)
        + tl.load(p19)
        + tl.load(p20)
        + tl.load(p21)
        + tl.load(p22)
        + tl.load(p23)
        + tl.load(p24)
        + tl.load(p25)
        + tl.load(p26)
        + tl.load(p27)
        + tl.load(p28)
        + tl.load(p29)
    )
    tl.store(p0, value)


@triton.jit(do_not_specialize=(0,))
def launch_constexpr_heavy_30(
    p0,
    c0: tl.constexpr,
    c1: tl.constexpr,
    c2: tl.constexpr,
    c3: tl.constexpr,
    c4: tl.constexpr,
    c5: tl.constexpr,
    c6: tl.constexpr,
    c7: tl.constexpr,
    c8: tl.constexpr,
    c9: tl.constexpr,
    c10: tl.constexpr,
    c11: tl.constexpr,
    c12: tl.constexpr,
    c13: tl.constexpr,
    c14: tl.constexpr,
    c15: tl.constexpr,
    c16: tl.constexpr,
    c17: tl.constexpr,
    c18: tl.constexpr,
    c19: tl.constexpr,
    c20: tl.constexpr,
    c21: tl.constexpr,
    c22: tl.constexpr,
    c23: tl.constexpr,
    c24: tl.constexpr,
    c25: tl.constexpr,
    c26: tl.constexpr,
    c27: tl.constexpr,
    c28: tl.constexpr,
):
    value = (
        c0
        + c1
        + c2
        + c3
        + c4
        + c5
        + c6
        + c7
        + c8
        + c9
        + c10
        + c11
        + c12
        + c13
        + c14
        + c15
        + c16
        + c17
        + c18
        + c19
        + c20
        + c21
        + c22
        + c23
        + c24
        + c25
        + c26
        + c27
        + c28
    )
    tl.store(p0, value)


@triton.jit(do_not_specialize=range(15))
def launch_mixed_30(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    c0: tl.constexpr,
    c1: tl.constexpr,
    c2: tl.constexpr,
    c3: tl.constexpr,
    c4: tl.constexpr,
    c5: tl.constexpr,
    c6: tl.constexpr,
    c7: tl.constexpr,
    c8: tl.constexpr,
    c9: tl.constexpr,
    c10: tl.constexpr,
    c11: tl.constexpr,
    c12: tl.constexpr,
    c13: tl.constexpr,
    c14: tl.constexpr,
):
    value = (
        tl.load(p1)
        + tl.load(p2)
        + tl.load(p3)
        + tl.load(p4)
        + tl.load(p5)
        + tl.load(p6)
        + tl.load(p7)
        + tl.load(p8)
        + tl.load(p9)
        + tl.load(p10)
        + tl.load(p11)
        + tl.load(p12)
        + tl.load(p13)
        + tl.load(p14)
        + c0
        + c1
        + c2
        + c3
        + c4
        + c5
        + c6
        + c7
        + c8
        + c9
        + c10
        + c11
        + c12
        + c13
        + c14
    )
    tl.store(p0, value)
