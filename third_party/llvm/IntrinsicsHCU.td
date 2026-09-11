// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd., 2026.

//===- IntrinsicsHCU.td - Defines HCU intrinsics -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the HCU-specific intrinsics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// HCU intrinsics
// ===----------------------------------------------------------------------===//

// Used to tag image and resource intrinsics with information used to generate
// mem operands.
class HCURsrcIntrinsicBase<int rsrcarg, bit isimage = false> {
  int RsrcArg = rsrcarg;
  bit IsImage = isimage;
}


let TargetPrefix = "hcu" in {

// legacy intrinsics
def int_hcu_fma_mix_f32 :
  // GCCBuiltin<"__builtin_hcu_fma_mix_f32">,
  Intrinsic<[llvm_float_ty],
            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty],
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

// gfx926 intrinsics
class HCUMMopVstepIntrinsic<LLVMType DestTy, LLVMType SrcABTy> :
  Intrinsic<[DestTy],
            [SrcABTy, SrcABTy, DestTy, llvm_i32_ty, llvm_i1_ty],
            [IntrConvergent, IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>]>;

class HCUMMopIntrinsic<LLVMType DestTy, LLVMType SrcABTy> :
  // GCCBuiltin<!subst("int", "__builtin", NAME)>,
  Intrinsic<[DestTy],
            [SrcABTy, SrcABTy, DestTy, llvm_i1_ty, llvm_i1_ty],
            [IntrConvergent, IntrNoMem, IntrWillReturn]>;

class HCUMMopClampIntrinsic<LLVMType DestTy, LLVMType SrcABTy> :
  Intrinsic<[DestTy],
            [SrcABTy, SrcABTy, DestTy, llvm_i1_ty, llvm_i1_ty, llvm_i1_ty],
            [IntrConvergent, IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>]>;

def int_hcu_mmac_16x16x4_f64       : HCUMMopVstepIntrinsic<llvm_v4f64_ty, llvm_double_ty>;
def int_hcu_mmac_16x16x4_f32       : HCUMMopVstepIntrinsic<llvm_v4f32_ty, llvm_float_ty>;

def int_hcu_mmac_16x16x8_f32       : HCUMMopIntrinsic<llvm_v4f32_ty, llvm_v2f32_ty>;
def int_hcu_mmac_f32_16x16x8_tf32  : HCUMMopIntrinsic<llvm_v4f32_ty, llvm_v2i32_ty>;
def int_hcu_mmac_f32_16x16x16_f16  : HCUMMopIntrinsic<llvm_v4f32_ty, llvm_v4f16_ty>;
def int_hcu_mmac_f32_16x16x16_bf16 : HCUMMopIntrinsic<llvm_v4f32_ty, llvm_v4i16_ty>;

def int_hcu_mmac_i32_16x16x32_i8   : HCUMMopClampIntrinsic<llvm_v4i32_ty, llvm_v2i32_ty>;
def int_hcu_mmac_i32_16x16x32_u8   : HCUMMopClampIntrinsic<llvm_v4i32_ty, llvm_v2i32_ty>;
def int_hcu_mmac_i32_16x16x64_i4   : HCUMMopClampIntrinsic<llvm_v4i32_ty, llvm_v2i32_ty>;
def int_hcu_mmac_i32_16x16x64_u4   : HCUMMopClampIntrinsic<llvm_v4i32_ty, llvm_v2i32_ty>;

def int_hcu_mmac_f32_16x16x32_fp8_fp8   : HCUMMopIntrinsic<llvm_v4f32_ty, llvm_v2i32_ty>;
def int_hcu_mmac_f32_16x16x32_bf8_bf8   : HCUMMopIntrinsic<llvm_v4f32_ty, llvm_v2i32_ty>;
def int_hcu_mmac_f32_16x16x32_fp8_bf8   : HCUMMopIntrinsic<llvm_v4f32_ty, llvm_v2i32_ty>;
def int_hcu_mmac_f32_16x16x32_bf8_fp8   : HCUMMopIntrinsic<llvm_v4f32_ty, llvm_v2i32_ty>;

class HCUDsReadMatrixIntrinsic:
  Intrinsic<[llvm_anyvector_ty], [llvm_anyptr_ty, llvm_i32_ty],
    [IntrReadMem,IntrConvergent, IntrWillReturn,ImmArg<ArgIndex<1>>]>;

class HCUDsReadMatrixFormatIntrinsic:
  Intrinsic<[llvm_anyvector_ty], [llvm_anyptr_ty, llvm_i32_ty, llvm_i8_ty, llvm_i8_ty, llvm_i8_ty, llvm_i8_ty],
    [IntrReadMem,IntrConvergent, IntrWillReturn, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>,
     ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

class HCUDsWriteMatrixFormatIntrinsic:
  Intrinsic<[], [llvm_anyvector_ty, llvm_anyptr_ty, llvm_i32_ty, llvm_i8_ty, llvm_i8_ty, llvm_i8_ty, llvm_i8_ty, llvm_i1_ty],
    [IntrWriteMem,IntrConvergent, IntrWillReturn, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>,
     ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;

def int_hcu_ds_read_m32x64_b4  : HCUDsReadMatrixIntrinsic;
def int_hcu_ds_read_m32x32_b8  : HCUDsReadMatrixIntrinsic;
def int_hcu_ds_read_m32x16_b16 : HCUDsReadMatrixIntrinsic;
def int_hcu_ds_read_m32x16_b16_alt : HCUDsReadMatrixIntrinsic;
def int_hcu_ds_read_m32x8_b32  : HCUDsReadMatrixIntrinsic;
def int_hcu_ds_read_m32x32_b8_alt2 : HCUDsReadMatrixIntrinsic;
def int_hcu_ds_read_m64x16_b8_alt4 : HCUDsReadMatrixIntrinsic;
def int_hcu_ds_read_m16x16_b32 : HCUDsReadMatrixIntrinsic;
def int_hcu_ds_read_mt16x16_b32 : HCUDsReadMatrixIntrinsic;
def int_hcu_ds_read_mt16x32_b16_alt2 : HCUDsReadMatrixIntrinsic;

def int_hcu_ds_read_matrix_format : HCUDsReadMatrixFormatIntrinsic;
def int_hcu_ds_read_matrix_trans_format : HCUDsReadMatrixFormatIntrinsic;
def int_hcu_ds_write_matrix_format : HCUDsWriteMatrixFormatIntrinsic;

class HCUDsReadPackIntrinsic:
  Intrinsic<[llvm_i32_ty], [LLVMQualPointerType<3>, llvm_i32_ty],
    [IntrReadMem,IntrConvergent, IntrWillReturn]>;

def int_hcu_ds_read_pack_l_b16 : HCUDsReadPackIntrinsic;
def int_hcu_ds_read_pack_h_b16 : HCUDsReadPackIntrinsic;
def int_hcu_ds_read_pack_0_b8 : HCUDsReadPackIntrinsic;
def int_hcu_ds_read_pack_1_b8 : HCUDsReadPackIntrinsic;
def int_hcu_ds_read_pack_2_b8 : HCUDsReadPackIntrinsic;
def int_hcu_ds_read_pack_3_b8 : HCUDsReadPackIntrinsic;

def int_hcu_ds_read_mask_b32:
  Intrinsic<[llvm_i32_ty], [LLVMQualPointerType<3>, llvm_i32_ty, llvm_i32_ty],
    [IntrReadMem,IntrConvergent, IntrWillReturn]>;

// GFX938 conversion
// llvm.hcu.cvt.f32.bf8 float vdst, int srcA, imm clamp, imm omod, imm byte_sel [0..3]
// byte_sel selects byte from srcA.
def int_hcu_cvt_f32_bf8 :
  // GCCBuiltin<"__builtin_hcu_cvt_f32_bf8">,
  Intrinsic<[llvm_float_ty],
            [llvm_i32_ty, llvm_i1_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

// llvm.hcu.cvt.f32.fp8 float vdst, int srcA, imm clamp , imm omod, imm byte_sel [0..3]
def int_hcu_cvt_f32_fp8 :
  // GCCBuiltin<"__builtin_hcu_cvt_f32_fp8">,
  Intrinsic<[llvm_float_ty],
            [llvm_i32_ty, llvm_i1_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

// llvm.hcu.cvt.pk.f32.bf8 float2 vdst, int srcA, imm word_sel
// word_sel = 1 selects 2 high bytes, 0 selects 2 low bytes.
def int_hcu_cvt_pk_f32_bf8 :
  ClangBuiltin<"__builtin_hcu_cvt_pk_f32_bf8">,
  DefaultAttrsIntrinsic<[llvm_v2f32_ty],
            [llvm_i32_ty, llvm_i1_ty],
            [IntrNoMem, ImmArg<ArgIndex<1>>]>;

// llvm.hcu.cvt.pk.f32.fp8 float2 vdst, int srcA, imm word_sel.
def int_hcu_cvt_pk_f32_fp8 :
  ClangBuiltin<"__builtin_hcu_cvt_pk_f32_fp8">,
  DefaultAttrsIntrinsic<[llvm_v2f32_ty],
            [llvm_i32_ty, llvm_i1_ty],
            [IntrNoMem, ImmArg<ArgIndex<1>>]>;

// llvm.hcu.cvt.pk.bf8.f32 int vdst, float srcA, float srcB, int old, imm word_sel
// word_sel = 1 selects 2 high bytes in the vdst, 0 selects 2 low bytes.
def int_hcu_cvt_pk_bf8_f32 :
  // GCCBuiltin<"__builtin_hcu_cvt_pk_bf8_f32">,
  Intrinsic<[llvm_i32_ty],
            [llvm_float_ty, llvm_float_ty, llvm_i32_ty, llvm_i1_ty],
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>]>;

// llvm.hcu.cvt.pk.fp8.f32 int vdst, float srcA, float srcB, int old, imm word_sel
def int_hcu_cvt_pk_fp8_f32 :
  // GCCBuiltin<"__builtin_hcu_cvt_pk_fp8_f32">,
  Intrinsic<[llvm_i32_ty],
            [llvm_float_ty, llvm_float_ty, llvm_i32_ty, llvm_i1_ty],
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>]>;

// llvm.hcu.cvt.pk.f16.f32 halfx2 vdst, float srcA, float srcB, imm clamp, int omod
def int_hcu_cvt_pk_f16_f32_v1 :
  // GCCBuiltin<"__builtin_hcu_cvt_pk_f16_f32">,
  Intrinsic<[llvm_v2f16_ty],
            [llvm_float_ty, llvm_float_ty, llvm_i1_ty, llvm_i32_ty],
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

// llvm.hcu.cvt.pk.bf16.f32 shortx2 vdst, float srcA, float srcB, imm clamp
def int_hcu_cvt_pk_bf16_f32_v1 :
  // GCCBuiltin<"__builtin_hcu_cvt_pk_f16_f32">,
  Intrinsic<[llvm_v2i16_ty],
            [llvm_float_ty, llvm_float_ty, llvm_i1_ty],
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<2>>]>;

// llvm.hcu.cvt.pk.bf16.f32 float vdst, int src0_modifier, float src0, int src1_modifier, int src1, imm clamp
// Lowered from __builtin_hcu_cvt_pk_bf16_f32 in Clang (CGBuiltin); no ClangBuiltin
// mapping here so EmitBuiltinExpr does not use the generic intrinsic emitter (which
// cannot widen ICE literals to i1 for imm clamp, etc.).
def int_hcu_cvt_pk_bf16_f32 :
  DefaultAttrsIntrinsic<[llvm_v2bf16_ty],
            [llvm_i32_ty, llvm_float_ty, llvm_i32_ty, llvm_float_ty, llvm_i1_ty],
            [IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

// llvm.hcu.cvt.pk.f16.f32 float vdst, imm src0_modifier, float src0, imm src1_modifier, int src1, imm clamp, imm omod
// Same: emitted only via EmitAMDGPUBuiltinExpr, not the generic ClangBuiltin path.
def int_hcu_cvt_pk_f16_f32 :
  DefaultAttrsIntrinsic<[llvm_v2f16_ty],
            [llvm_i32_ty, llvm_float_ty, llvm_i32_ty, llvm_float_ty, llvm_i1_ty, llvm_i32_ty],
            [IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;


// llvm.hcu.cvt.sr.bf8.f32 int vdst, float srcA, int srcB, int old, imm byte_sel [0..3]
// byte_sel selects byte to write into vdst.
def int_hcu_cvt_sr_bf8_f32 :
  // GCCBuiltin<"__builtin_hcu_cvt_sr_bf8_f32">,
  Intrinsic<[llvm_i32_ty],
            [llvm_float_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>]>;

// llvm.hcu.cvt.sr.fp8.f32 int vdst, float srcA, int srcB, int old, imm byte_sel [0..3]
def int_hcu_cvt_sr_fp8_f32 :
  // GCCBuiltin<"__builtin_hcu_cvt_sr_fp8_f32">,
  Intrinsic<[llvm_i32_ty],
            [llvm_float_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>]>;

def int_hcu_cvt_f32_bf16 :
  // GCCBuiltin<"__builtin_hcu_cvt_f32_bf16">,
  Intrinsic<[llvm_float_ty],
            [llvm_i32_ty, llvm_i1_ty, llvm_i32_ty, llvm_i1_ty],
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_hcu_cvt_bf16_f32 :
  // GCCBuiltin<"__builtin_hcu_cvt_bf16_f32">,
  Intrinsic<[llvm_i16_ty],
            [llvm_float_ty, llvm_i1_ty, llvm_i1_ty],
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

class HCUVOP1InModClampIntrinsic<LLVMType DestTy, LLVMType SrcTy>:
  Intrinsic<[DestTy],
            [SrcTy, llvm_i1_ty],
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<1>>]>;

def int_hcu_cvt_rtz_tf32_f32    :  HCUVOP1InModClampIntrinsic<llvm_i32_ty, llvm_float_ty>;
def int_hcu_cvt_rna_tf32_f32    :  HCUVOP1InModClampIntrinsic<llvm_i32_ty, llvm_float_ty>;
def int_hcu_cvt_rtne_tf32_f32   :  HCUVOP1InModClampIntrinsic<llvm_i32_ty, llvm_float_ty>;

// VOP3
def int_hcu_pk_fma_f32:
  Intrinsic<[llvm_v2f32_ty],
            [llvm_v2f32_ty,
             llvm_v2f32_ty,
             llvm_v2f32_ty,
             llvm_vararg_ty], // op_sel: default value is 0 ([0,0])
                              // op_sel_hi: default value is 7 ([1,1,1])
            [IntrNoMem, IntrWillReturn]>;

def int_hcu_pk_add_f32:
  Intrinsic<[llvm_v2f32_ty],
            [llvm_v2f32_ty,
             llvm_v2f32_ty,
             llvm_vararg_ty], // op_sel: default value is 0 ([0,0])
                              // op_sel_hi: default value is 3 ([1,1])
            [IntrNoMem, IntrWillReturn]>;

def int_hcu_pk_mul_f32:
  Intrinsic<[llvm_v2f32_ty],
            [llvm_v2f32_ty,
             llvm_v2f32_ty,
             llvm_vararg_ty], // op_sel: default value is 0 ([0,0])
                              // op_sel_hi: default value is 3 ([1,1])
            [IntrNoMem, IntrWillReturn]>;



//HCU: buffer load to lds
defset list<HCURsrcIntrinsicBase> HCUBufferIntrinsics = {
class HCUBufferLoadLDS :
  // GCCBuiltin<!subst("int", "__builtin", NAME)>,
  Intrinsic <
  [],
  [llvm_v4i32_ty,                      // rsrc(SGPR)
   LLVMQualPointerType<3>, // LDS base offset(imm)
   llvm_i32_ty,                        // Data byte size: 4
   llvm_i32_ty,                        // vindex(VGPR)
   llvm_i32_ty,                        // voffset(VGPR, included in bounds checking and swizzling)
   llvm_i32_ty,                        // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty,                        // imm offset(imm, included in bounds checking and swizzling)
   llvm_i32_ty],                       // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                                       //                                       bit 1 = slc,
                                       //                                       bit 2 = dlc on gfx10+))
                                       //                      swizzled buffer (bit 3 = swz))
  [IntrWillReturn, NoCapture<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<6>>,
   ImmArg<ArgIndex<7>>], "", [SDNPMemOperand]>, HCURsrcIntrinsicBase<0>;
def int_hcu_buffer_load_lds : HCUBufferLoadLDS;

class HCURawBufferLoadLDS :
  // GCCBuiltin<!subst("int", "__builtin", NAME)>,
  Intrinsic <
  [],
  [llvm_v4i32_ty,                      // rsrc(SGPR)
   LLVMQualPointerType<3>, // LDS base offset(imm)
   llvm_i32_ty,                        // Data byte size: 4
   llvm_i32_ty,                        // voffset(VGPR, included in bounds checking and swizzling)
   llvm_i32_ty,                        // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty,                        // imm offset(imm, included in bounds checking and swizzling)
   llvm_i32_ty],                       // auxiliary data (imm, cachepolicy     (bit 0 = glc,
                                       //                                       bit 1 = slc,
                                       //                                       bit 2 = dlc on gfx10+))
                                       //                      swizzled buffer (bit 3 = swz))
  [IntrWillReturn, NoCapture<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<5>>,
   ImmArg<ArgIndex<6>>], "", [SDNPMemOperand]>, HCURsrcIntrinsicBase<0>;
def int_hcu_raw_buffer_load_lds : HCURawBufferLoadLDS;
} // defset HCUBufferIntrinsics


//===----------------------------------------------------------------------===//
// gfx936 intrinsics
// ===----------------------------------------------------------------------===//
def int_hcu_mov64 : Intrinsic<[llvm_i64_ty], [llvm_i64_ty],
  [IntrNoMem, IntrSpeculatable]>;
// matrix_load intrinsics
class HCUMatrixLoadLDS :
  Intrinsic <
  [],                                  // no return (void)
  [llvm_v4i32_ty,                      // rsrc(SGPR)
   llvm_anyptr_ty,                     // soffset (SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty,                        // offset (SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i1_ty,                        // t
   llvm_i1_ty,                        // r
   llvm_i1_ty,                        // glc
   llvm_i1_ty,                        // slc
   llvm_i1_ty],                       // bps
  [IntrWillReturn, IntrWriteMem, IntrConvergent, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>,
    ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;


def int_hcu_matrix_load_64x16_b8 : HCUMatrixLoadLDS;
def int_hcu_matrix_load_128x16_b8 : HCUMatrixLoadLDS;
def int_hcu_matrix_load_64x32_b8 : HCUMatrixLoadLDS;
def int_hcu_matrix_load_32x16_b16 : HCUMatrixLoadLDS;
def int_hcu_matrix_load_64x16_b16 : HCUMatrixLoadLDS;
def int_hcu_matrix_load_32x32_b16 : HCUMatrixLoadLDS;

def int_hcu_matrix_load_16x16_b32 : HCUMatrixLoadLDS;
def int_hcu_matrix_load_64x16_b6 : HCUMatrixLoadLDS;
def int_hcu_matrix_load_64x16_u4 : HCUMatrixLoadLDS;
def int_hcu_matrix_load_64x16_i4 : HCUMatrixLoadLDS;
def int_hcu_matrix_load_64x16_fp4 : HCUMatrixLoadLDS;
def int_hcu_matrix_load_128x16_b4 : HCUMatrixLoadLDS;
def int_hcu_matrix_load_256x16_b4 : HCUMatrixLoadLDS;

class HCUMatrixStoreLDS :
  Intrinsic <
  [],                                  // no return (void)
  [llvm_v4i32_ty,                      // rsrc(SGPR)
   llvm_anyptr_ty,                     // soffset (SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty,                        // offset (SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i1_ty,                         // t
   llvm_i1_ty,                         // r
   llvm_i1_ty,                         // glc
   llvm_i1_ty],                        // slc
  [IntrWillReturn, IntrWriteMem, IntrConvergent, ImmArg<ArgIndex<2>>,
   ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>,
   ImmArg<ArgIndex<6>>]>;

def int_hcu_matrix_store_128x16_b4 : HCUMatrixStoreLDS;
def int_hcu_matrix_store_64x16_b8 : HCUMatrixStoreLDS;
def int_hcu_matrix_store_32x16_b16 : HCUMatrixStoreLDS;
def int_hcu_matrix_store_64x16_b6 : HCUMatrixStoreLDS;
def int_hcu_matrix_store_16x16_b32 : HCUMatrixStoreLDS;
def int_hcu_matrix_store_pk_64x16_b4 : HCUMatrixStoreLDS;
def int_hcu_matrix_store_64x32_b8 : HCUMatrixStoreLDS;
def int_hcu_matrix_store_32x32_b16 : HCUMatrixStoreLDS;
def int_hcu_matrix_store_256x16_b4 : HCUMatrixStoreLDS;
def int_hcu_matrix_store_128x16_b8 : HCUMatrixStoreLDS;
def int_hcu_matrix_store_64x16_b16 : HCUMatrixStoreLDS;

class HCUMatrixStoreRTN :
  Intrinsic <
  [llvm_v4i32_ty],                     // writeback vdata
  [llvm_v4i32_ty,                      // rsrc(SGPR)
   llvm_anyptr_ty,                     // soffset (SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty,                        // offset (SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i1_ty,                         // t
   llvm_i1_ty,                         // r
   llvm_i1_ty,                         // glc
   llvm_i1_ty],                        // slc
  [IntrWillReturn, IntrConvergent, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>,
   ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>],
  "", [SDNPMemOperand]>;

def int_hcu_matrix_store_128x16_b4_rtn : HCUMatrixStoreRTN;
def int_hcu_matrix_store_64x16_b8_rtn : HCUMatrixStoreRTN;
def int_hcu_matrix_store_32x16_b16_rtn : HCUMatrixStoreRTN;
def int_hcu_matrix_store_64x16_b6_rtn : HCUMatrixStoreRTN;
def int_hcu_matrix_store_16x16_b32_rtn : HCUMatrixStoreRTN;
def int_hcu_matrix_store_pk_64x16_b4_rtn : HCUMatrixStoreRTN;
def int_hcu_matrix_store_64x32_b8_rtn : HCUMatrixStoreRTN;
def int_hcu_matrix_store_32x32_b16_rtn : HCUMatrixStoreRTN;


class AMDGPUDsRead2Intrinsic :
  Intrinsic<[llvm_anyvector_ty], [llvm_anyptr_ty, llvm_i8_ty, llvm_i8_ty, llvm_i1_ty],
            [IntrReadMem, IntrConvergent, IntrWillReturn, NoCapture<ArgIndex<0>>,
             ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_hcu_ds_read2_b64 : AMDGPUDsRead2Intrinsic;
def int_hcu_ds_read2_b32 : AMDGPUDsRead2Intrinsic;

// matrix_load intrinsics for b8/b16
class HCUMMCLDS_NO_RET :
  Intrinsic <
  [llvm_v4i32_ty],                                  // no return (void)
  [llvm_v2i32_ty,                      // rsrc(SGPR)
  llvm_anyptr_ty,
   llvm_i32_ty,                        // stride (SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty,                        // offset (SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i1_ty,                        // t
   llvm_i1_ty,                        // r
   llvm_i1_ty,                        // sw
   llvm_i1_ty,                        // glc
   llvm_i1_ty],                       // slc

  [IntrWillReturn, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>,
      ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>, ImmArg<ArgIndex<8>>],
       "", [SDNPMemOperand]>;
class HCUMMC :
  Intrinsic <
  [llvm_v4i32_ty],                                  // no return (void)
  [llvm_v2i32_ty,                      // rsrc(SGPR)
   llvm_i32_ty,                        // stride (SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty,                        // offset (SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i1_ty,                        // t
   llvm_i1_ty,                        // r
   llvm_i1_ty,                        // sw
   llvm_i1_ty,                        // glc
   llvm_i1_ty],                       // slc

  [IntrWillReturn, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>,
      ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>],
       "", [SDNPMemOperand]>;

def int_hcu_matrix_load_b8_lds : HCUMMCLDS_NO_RET;
def int_hcu_matrix_load_b16_lds : HCUMMCLDS_NO_RET;
def int_hcu_matrix_load_b8 : HCUMMC;
def int_hcu_matrix_load_b16 : HCUMMC;

//===----------------------------------------------------------------------===//
// gfx946 intrinsics
// ===----------------------------------------------------------------------===//

// ebarrier intrinsics
class EBarrierIntrinsic : Intrinsic<
  [],
  [llvm_i32_ty,  // ebar_id
   llvm_i32_ty], // wv_cnt or ebar_obj(slot_wr)
  [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

class EBarrierRedIntrinsic : Intrinsic<
  [llvm_i32_ty], // dst
  [llvm_i32_ty,  // ebar_id
   llvm_i32_ty], // wv_cnt
  [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

def int_hcu_s_ebarrier_arrive  : EBarrierIntrinsic;
def int_hcu_s_ebarrier_sync    : EBarrierIntrinsic;
def int_hcu_s_ebarrier_slot_wr : EBarrierIntrinsic;
def int_hcu_s_ebarrier_and     : EBarrierRedIntrinsic;
def int_hcu_s_ebarrier_or      : EBarrierRedIntrinsic;
def int_hcu_s_ebarrier_popc    : EBarrierRedIntrinsic;

def int_hcu_s_ebarrier_slot_rd : Intrinsic<
  [llvm_i32_ty], // ebar_obj
  [llvm_i32_ty], // ebar_id
  [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

// abarrier intrinsics
def int_hcu_s_abarrier_inv : Intrinsic<
  [],
  [llvm_i32_ty], // abar_id
  [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

def int_hcu_s_abarrier_seq : Intrinsic<
  [],
  [llvm_i32_ty], // abar_id
  [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

class ABarrierIntrinsic1Output2Input : Intrinsic<
  [llvm_i32_ty], // dst
  [llvm_i32_ty,  // abar_id
   llvm_vararg_ty], // wv_cnt or state (optional; backend may encode off)
  [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

def int_hcu_s_abarrier_arrive                 : ABarrierIntrinsic1Output2Input;
def int_hcu_s_abarrier_arrive_nocomplete      : ABarrierIntrinsic1Output2Input;
def int_hcu_s_abarrier_arrive_drop            : ABarrierIntrinsic1Output2Input;
def int_hcu_s_abarrier_arrive_drop_nocomplete : ABarrierIntrinsic1Output2Input;
def int_hcu_s_abarrier_try_wait               : ABarrierIntrinsic1Output2Input;
def int_hcu_s_abarrier_test_wait              : ABarrierIntrinsic1Output2Input;

class ABarrierIntrinsicNoOutput2Input : Intrinsic<
  [],
  [llvm_i32_ty,  // abar_id
   llvm_i32_ty], // cnt
  [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

def int_hcu_s_abarrier_init        : ABarrierIntrinsicNoOutput2Input;
def int_hcu_s_abarrier_expect_tx   : ABarrierIntrinsicNoOutput2Input;
def int_hcu_s_abarrier_complete_tx : ABarrierIntrinsicNoOutput2Input;

def int_hcu_s_abarrier_slot_rd : Intrinsic<
  [llvm_i32_ty], // abar_obj
  [llvm_i32_ty], // abar_id
  [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

def int_hcu_s_abarrier_slot_wr : Intrinsic<
  [],
  [llvm_i32_ty,  // abar_id
   llvm_i32_ty], // abar_obj
  [IntrNoMem, IntrHasSideEffects, IntrConvergent, IntrWillReturn, IntrNoCallback, IntrNoFree]
>;

def int_hcu_s_set_vgpr_size :
  Intrinsic<[], [llvm_i16_ty], [ImmArg<ArgIndex<0>>, IntrNoMem, IntrHasSideEffects, IntrConvergent,
                                IntrWillReturn, IntrNoCallback, IntrNoFree]>;

// mmac intrinsics
/*
  RealABTypeId:
    0:  fp8_fp8   1:  fp8_bf8   2:  fp8_fp6   3:  fp8_bf6   4:  fp8_fp4
    5:  bf8_fp8   6:  bf8_bf8   7:  bf8_fp6   8:  bf8_bf6   9:  bf8_fp4
    10: fp6_fp8   11: fp6_bf8   12: fp6_fp6   13: fp6_bf6   14: fp6_fp4
    15: bf6_fp8   16: bf6_bf8   17: bf6_fp6   18: bf6_bf6   19: bf6_fp4
    20: fp4_fp8   21: fp4_bf8   22: fp4_fp6   23: fp4_bf6   24: fp4_fp4
*/
class HCUMMopF8F6F4Intrinsic<LLVMType DestTy, LLVMType SrcABTy> :
  // GCCBuiltin<!subst("int", "__builtin", NAME)>,
  Intrinsic<[DestTy],
            [SrcABTy,
             SrcABTy,
             DestTy,
             llvm_i32_ty, // RealABTypeId: enumeration values of the real data types of AType_BType
             llvm_i1_ty,  // lit
             llvm_i1_ty], // lts
            [IntrConvergent, IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

// The RealABTypeId cannot be 0, 1, 5, or 6 for int_hcu_mmac_f32_16x16x32_f8f6f4,
// as v_mmac_f32_16x16x32_fp8/bf8_fp8/bf8 follow the old definition.
def int_hcu_mmac_f32_16x16x32_f8f6f4  : HCUMMopF8F6F4Intrinsic<llvm_v4f32_ty, llvm_v2i32_ty>;
def int_hcu_mmac_f16_16x16x32_f8f6f4  : HCUMMopF8F6F4Intrinsic<llvm_v4f16_ty, llvm_v2i32_ty>;
def int_hcu_mmac_bf16_16x16x32_f8f6f4 : HCUMMopF8F6F4Intrinsic<llvm_v4i16_ty, llvm_v2i32_ty>;

def int_hcu_mmac_16x16x16_f16      : HCUMMopIntrinsic<llvm_v4f16_ty, llvm_v4f16_ty>;
def int_hcu_mmac_16x16x16_bf16     : HCUMMopIntrinsic<llvm_v4i16_ty, llvm_v4i16_ty>;
def int_hcu_mmac_bf16_16x16x16_f16 : HCUMMopIntrinsic<llvm_v4i16_ty, llvm_v4f16_ty>;
def int_hcu_mmac_f16_16x16x16_bf16 : HCUMMopIntrinsic<llvm_v4f16_ty, llvm_v4i16_ty>;

def int_hcu_mmac_f32_16x16x64_fp4  : HCUMMopIntrinsic<llvm_v4f32_ty, llvm_v2i32_ty>;
def int_hcu_mmac_f16_16x16x64_fp4  : HCUMMopIntrinsic<llvm_v4f16_ty, llvm_v2i32_ty>;
def int_hcu_mmac_bf16_16x16x64_fp4 : HCUMMopIntrinsic<llvm_v4i16_ty, llvm_v2i32_ty>;

def int_hcu_mmac_scale_f32_16x16x32_f8f6f4 : Intrinsic<
  [llvm_v4f32_ty], // D
  [llvm_v2i32_ty,  // A
   llvm_v2i32_ty,  // B
   llvm_v4f32_ty,  // C
   llvm_i32_ty,    // RealABTypeId
   llvm_i1_ty,     // lit
   llvm_i1_ty,     // lts
   llvm_i32_ty,    // scale_a_offset
   llvm_i32_ty,],  // scale_b_offset
  [IntrConvergent, IntrNoMem, IntrWillReturn]>;

def int_hcu_mmac_scale_f32_16x16x64_fp4 : Intrinsic<
  [llvm_v4f32_ty], // D
  [llvm_v2i32_ty,  // A
   llvm_v2i32_ty,  // B
   llvm_v4f32_ty,  // C
   llvm_i1_ty,     // lit
   llvm_i1_ty,     // lts
   llvm_i32_ty,    // scale_a_offset
   llvm_i32_ty,],  // scale_b_offset
  [IntrConvergent, IntrNoMem, IntrWillReturn]>;

def int_hcu_ds_mpermute_b64 : Intrinsic<
  [llvm_v2i32_ty],
  [llvm_v2i32_ty,
   llvm_i8_ty],
  [IntrConvergent, IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<1>>]>;

class HCUDsScaleCopyIntrinsic:
  Intrinsic<[],
            [LLVMQualPointerType<3>,
             llvm_i32_ty,
             llvm_i8_ty,
             llvm_i8_ty,
             llvm_i8_ty],
            [IntrConvergent, IntrWillReturn, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_hcu_ds_scale_copy_ds2buf : HCUDsScaleCopyIntrinsic;
def int_hcu_ds_scale_copy_buf2ds : HCUDsScaleCopyIntrinsic;

def int_hcu_ds_read_matrix_padbyte : HCUDsReadMatrixFormatIntrinsic;
def int_hcu_ds_read_matrix_trans_padbyte : HCUDsReadMatrixFormatIntrinsic;

class HCUDsPkComputeIntrinsic<LLVMType MemType, string builtin_name, bit IsCmp = 0> :
  ClangBuiltin<builtin_name>,
  DefaultAttrsIntrinsic<[MemType],
                        !if(IsCmp, [LLVMQualPointerType<3>, MemType, MemType], [LLVMQualPointerType<3>, MemType]),
                        [IntrArgMemOnly, NoCapture<ArgIndex<0>>], "",
                        [SDNPMayLoad, SDNPMayStore, SDNPMemOperand, SDNPHasChain]>;

def int_hcu_ds_fadd_v2f16 : HCUDsPkComputeIntrinsic<llvm_v2f16_ty, "__builtin_hcu_ds_atomic_fadd_v2f16">;
def int_hcu_ds_fmin_v2f16 : HCUDsPkComputeIntrinsic<llvm_v2f16_ty, "__builtin_hcu_ds_atomic_fmin_v2f16">;
def int_hcu_ds_fmax_v2f16 : HCUDsPkComputeIntrinsic<llvm_v2f16_ty, "__builtin_hcu_ds_atomic_fmax_v2f16">;
def int_hcu_ds_fcmpst_v2f16 : HCUDsPkComputeIntrinsic<llvm_v2f16_ty, "__builtin_hcu_ds_atomic_fcmpst_v2f16", 1>;
def int_hcu_ds_fadd_v2bf16 : HCUDsPkComputeIntrinsic<llvm_v2i16_ty, "__builtin_hcu_ds_atomic_fadd_v2bf16">;
def int_hcu_ds_fmin_v2bf16 : HCUDsPkComputeIntrinsic<llvm_v2i16_ty, "__builtin_hcu_ds_atomic_fmin_v2bf16">;
def int_hcu_ds_fmax_v2bf16 : HCUDsPkComputeIntrinsic<llvm_v2i16_ty, "__builtin_hcu_ds_atomic_fmax_v2bf16">;
def int_hcu_ds_fcmpst_v2bf16 : HCUDsPkComputeIntrinsic<llvm_v2i16_ty, "__builtin_hcu_ds_atomic_fcmpst_v2bf16", 1>;

// VOP1
def int_hcu_prng_b32 :
  ClangBuiltin<"__builtin_hcu_prng_b32">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrWillReturn]>;

class HCUPermlaneSwap :
  Intrinsic<[llvm_i32_ty,  // dst
             llvm_i32_ty], // src0_out
            [llvm_i32_ty,  // dst_in
             llvm_i32_ty,  // src
             llvm_i1_ty,   // bound_ctrl
             llvm_i1_ty],  // fi
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_hcu_permlane16_swap_b32 :
  // ClangBuiltin<"__builtin_hcu_permlane16_swap_b32">,
  HCUPermlaneSwap;

def int_hcu_permlane32_swap_b32 :
  // ClangBuiltin<"__builtin_hcu_permlane32_swap_b32">,
  HCUPermlaneSwap;

def int_hcu_cvt_pk_f32_e8m0 :
  ClangBuiltin<"__builtin_hcu_cvt_pk_f32_e8m0">,
  Intrinsic<[llvm_v2f32_ty], // dst
            [llvm_i32_ty,    // src
             llvm_i1_ty,     // clamp
             llvm_i32_ty,    // omod
             llvm_i1_ty],    // src0_sel(0:selects 2 low bytes, 1:selects 2 high bytes)
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_hcu_cvt_pk_bf16_e8m0 :
  ClangBuiltin<"__builtin_hcu_cvt_pk_bf16_e8m0">,
  Intrinsic<[llvm_v2i16_ty], // dst
            [llvm_i32_ty,    // src
             llvm_i1_ty,     // clamp
             llvm_i1_ty],    // src0_sel(0:selects 2 low bytes, 1:selects 2 high bytes)
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_hcu_cvt_pkrtz_e8m0_bf16 :
  ClangBuiltin<"__builtin_hcu_cvt_pkrtz_e8m0_bf16">,
  Intrinsic<[llvm_i32_ty],  // dst
            [llvm_v2i16_ty, // src
             llvm_i1_ty],   // dst_sel (0:selects 2 low bytes, 1:selects 2 high bytes)
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<1>>]>;

def int_hcu_cvt_pkrp_e8m0_bf16 :
  ClangBuiltin<"__builtin_hcu_cvt_pkrp_e8m0_bf16">,
  Intrinsic<[llvm_i32_ty],  // dst
            [llvm_v2i16_ty, // src
             llvm_i1_ty],   // dst_sel (0:selects 2 low bytes, 1:selects 2 high bytes)
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<1>>]>;

// VOP3
class HCUCvtSrPkF16FromF32Intrinsic<LLVMType DstTy, string builtin_name> :
  ClangBuiltin<builtin_name>,
  Intrinsic<[DstTy],         // dst
            [llvm_float_ty,  // src0
             llvm_float_ty,  // src1
             llvm_v2i16_ty], // two-channel seed data
            [IntrNoMem, IntrWillReturn]>;

def int_hcu_cvt_sr_pk_f16_f32  : HCUCvtSrPkF16FromF32Intrinsic<llvm_v2f16_ty, "__builtin_hcu_cvt_sr_pk_f16_f32">;
def int_hcu_cvt_sr_pk_bf16_f32 : HCUCvtSrPkF16FromF32Intrinsic<llvm_v2i16_ty, "__builtin_hcu_cvt_sr_pk_bf16_f32">;

def int_hcu_cvt_pk_i8_f32 :
  ClangBuiltin<"__builtin_hcu_cvt_pk_i8_f32">,
  Intrinsic<[llvm_i32_ty],  // dst
            [llvm_float_ty, // src0
             llvm_float_ty, // src1
             llvm_i32_ty,   // src2: other bits, dst[x] = src2.b32[x]
             llvm_i1_ty],   // op_sel for dst (0:selects 2 low bytes, 1:selects 2 high bytes)
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>]>;

def int_hcu_lop3_b32 :
  ClangBuiltin<"__builtin_hcu_lop3_b32">,
  Intrinsic<[llvm_i32_ty], // dst
            [llvm_i32_ty,  // src0
             llvm_i32_ty,  // src1
             llvm_i32_ty,  // src2
             llvm_i8_ty],  // lut_op
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>]>;

class HCUCvtScaleF32FromF8Intrinsic<string builtin_name> :
  ClangBuiltin<builtin_name>,
  Intrinsic<[llvm_float_ty], // dst
            [llvm_i32_ty,    // src0
             llvm_i32_ty,    // scale data
             llvm_i32_ty,    // op_sel for src0 (select byte from src0 [0..3])
             llvm_i1_ty,     // clamp
             llvm_i32_ty],   // scale_sel (select byte from scale data [0..3])
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_hcu_cvt_scale_f32_fp8 : HCUCvtScaleF32FromF8Intrinsic<"__builtin_hcu_cvt_scale_f32_fp8">;
def int_hcu_cvt_scale_f32_bf8 : HCUCvtScaleF32FromF8Intrinsic<"__builtin_hcu_cvt_scale_f32_bf8">;

class HCUCvtScaleF16FromF8Intrinsic<string builtin_name> :
  ClangBuiltin<builtin_name>,
  Intrinsic<[llvm_half_ty], // dst
            [llvm_i32_ty,   // src0
             llvm_i32_ty,   // scale data
             llvm_i32_ty,   // src2: other bits, dst[x] = src2.b32[x]
             llvm_i32_ty,   // select byte from src0 ([0..3])
             llvm_i1_ty,    // op_sel for dst (0:selects 2 low bytes, 1:selects 2 high bytes)
             llvm_i1_ty,    // clamp
             llvm_i32_ty],  // scale_sel for scale data (select byte [0..3])
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_hcu_cvt_scale_f16_fp8 : HCUCvtScaleF16FromF8Intrinsic<"__builtin_hcu_cvt_scale_f16_fp8">;
def int_hcu_cvt_scale_f16_bf8 : HCUCvtScaleF16FromF8Intrinsic<"__builtin_hcu_cvt_scale_f16_bf8">;

class HCUCvtScalePkF8FromF32Intrinsic<string builtin_name> :
  ClangBuiltin<builtin_name>,
  Intrinsic<[llvm_i32_ty],  // dst
            [llvm_float_ty, // src0
             llvm_float_ty, // src1
             llvm_i32_ty,   // scale data
             llvm_i1_ty,    // op_sel for dst (0:selects 2 low bytes, 1:selects 2 high bytes)
             llvm_i32_ty],  // scale_sel for scale data (select byte [0..3])
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_hcu_cvt_scale_pk_fp8_f32 : HCUCvtScalePkF8FromF32Intrinsic<"__builtin_hcu_cvt_scale_pk_fp8_f32">;
def int_hcu_cvt_scale_pk_bf8_f32 : HCUCvtScalePkF8FromF32Intrinsic<"__builtin_hcu_cvt_scale_pk_bf8_f32">;

class HCUCvtScalePkF6F4FromF32Intrinsic<string builtin_name> :
  ClangBuiltin<builtin_name>,
  Intrinsic<[llvm_i32_ty],  // dst
            [llvm_float_ty, // src0
             llvm_float_ty, // src1
             llvm_i32_ty,   // scale data
             llvm_i32_ty,   // op_sel for dst
             llvm_i32_ty],  // scale_sel for scale data (select byte [0..3])
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_hcu_cvt_scale_pk_fp6_f32 : HCUCvtScalePkF6F4FromF32Intrinsic<"__builtin_hcu_cvt_scale_pk_fp6_f32">;
def int_hcu_cvt_scale_pk_bf6_f32 : HCUCvtScalePkF6F4FromF32Intrinsic<"__builtin_hcu_cvt_scale_pk_bf6_f32">;
def int_hcu_cvt_scale_pk_fp4_f32 : HCUCvtScalePkF6F4FromF32Intrinsic<"__builtin_hcu_cvt_scale_pk_fp4_f32">;

class HCUCvtScalePkF8FromF16Intrinsic<LLVMType Src0Ty, string builtin_name> :
  ClangBuiltin<builtin_name>,
  Intrinsic<[llvm_i32_ty],  // dst
            [Src0Ty,       // src0
             llvm_i32_ty,  // scale data
             llvm_i32_ty,  // src2: other bits, dst[x] = src2.b32[x]
             llvm_i1_ty,   // op_sel for dst (0:selects 2 low bytes, 1:selects 2 high bytes)
             llvm_i32_ty], // scale_sel for scale data (select byte [0..3])
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_hcu_cvt_scale_pk_fp8_f16  : HCUCvtScalePkF8FromF16Intrinsic<llvm_v2f16_ty, "__builtin_hcu_cvt_scale_pk_fp8_f16">;
def int_hcu_cvt_scale_pk_bf8_f16  : HCUCvtScalePkF8FromF16Intrinsic<llvm_v2f16_ty, "__builtin_hcu_cvt_scale_pk_bf8_f16">;
def int_hcu_cvt_scale_pk_fp8_bf16 : HCUCvtScalePkF8FromF16Intrinsic<llvm_v2i16_ty, "__builtin_hcu_cvt_scale_pk_fp8_bf16">;
def int_hcu_cvt_scale_pk_bf8_bf16 : HCUCvtScalePkF8FromF16Intrinsic<llvm_v2i16_ty, "__builtin_hcu_cvt_scale_pk_bf8_bf16">;

class HCUCvtScalePkF6F4FromF16Intrinsic<LLVMType Src0Ty, string builtin_name> :
  ClangBuiltin<builtin_name>,
  Intrinsic<[llvm_i32_ty],  // dst
            [Src0Ty,       // src0
             llvm_i32_ty,  // scale data
             llvm_i32_ty,  // src2: other bits, dst[x] = src2.b32[x]
             llvm_i32_ty,  // op_sel for dst
             llvm_i32_ty], // scale_sel for scale data (select byte [0..3])
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_hcu_cvt_scale_pk_fp6_f16  : HCUCvtScalePkF6F4FromF16Intrinsic<llvm_v2f16_ty, "__builtin_hcu_cvt_scale_pk_fp6_f16">;
def int_hcu_cvt_scale_pk_bf6_f16  : HCUCvtScalePkF6F4FromF16Intrinsic<llvm_v2f16_ty, "__builtin_hcu_cvt_scale_pk_bf6_f16">;
def int_hcu_cvt_scale_pk_fp4_f16  : HCUCvtScalePkF6F4FromF16Intrinsic<llvm_v2f16_ty, "__builtin_hcu_cvt_scale_pk_fp4_f16">;
def int_hcu_cvt_scale_pk_fp6_bf16 : HCUCvtScalePkF6F4FromF16Intrinsic<llvm_v2i16_ty, "__builtin_hcu_cvt_scale_pk_fp6_bf16">;
def int_hcu_cvt_scale_pk_bf6_bf16 : HCUCvtScalePkF6F4FromF16Intrinsic<llvm_v2i16_ty, "__builtin_hcu_cvt_scale_pk_bf6_bf16">;
def int_hcu_cvt_scale_pk_fp4_bf16 : HCUCvtScalePkF6F4FromF16Intrinsic<llvm_v2i16_ty, "__builtin_hcu_cvt_scale_pk_fp4_bf16">;

class HCUCvtScalePkF32F16FromF8F6Intrinsic<LLVMType DstTy, string builtin_name> :
  ClangBuiltin<builtin_name>,
  Intrinsic<[DstTy],       // dst
            [llvm_i32_ty,  // src0
             llvm_i32_ty,  // scale data
             llvm_i1_ty,   // op_sel for src0 (0:selects 2 low bytes, 1:selects 2 high bytes)
             llvm_i1_ty,   // clamp
             llvm_i32_ty], // scale_sel for scale data (select byte [0..3])
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_hcu_cvt_scale_pk_f32_fp8  : HCUCvtScalePkF32F16FromF8F6Intrinsic<llvm_v2f32_ty, "__builtin_hcu_cvt_scale_pk_f32_fp8">;
def int_hcu_cvt_scale_pk_f32_bf8  : HCUCvtScalePkF32F16FromF8F6Intrinsic<llvm_v2f32_ty, "__builtin_hcu_cvt_scale_pk_f32_bf8">;
def int_hcu_cvt_scale_pk_f32_fp6  : HCUCvtScalePkF32F16FromF8F6Intrinsic<llvm_v2f32_ty, "__builtin_hcu_cvt_scale_pk_f32_fp6">;
def int_hcu_cvt_scale_pk_f32_bf6  : HCUCvtScalePkF32F16FromF8F6Intrinsic<llvm_v2f32_ty, "__builtin_hcu_cvt_scale_pk_f32_bf6">;
def int_hcu_cvt_scale_pk_f16_fp8  : HCUCvtScalePkF32F16FromF8F6Intrinsic<llvm_v2f16_ty, "__builtin_hcu_cvt_scale_pk_f16_fp8">;
def int_hcu_cvt_scale_pk_f16_bf8  : HCUCvtScalePkF32F16FromF8F6Intrinsic<llvm_v2f16_ty, "__builtin_hcu_cvt_scale_pk_f16_bf8">;
def int_hcu_cvt_scale_pk_f16_fp6  : HCUCvtScalePkF32F16FromF8F6Intrinsic<llvm_v2f16_ty, "__builtin_hcu_cvt_scale_pk_f16_fp6">;
def int_hcu_cvt_scale_pk_f16_bf6  : HCUCvtScalePkF32F16FromF8F6Intrinsic<llvm_v2f16_ty, "__builtin_hcu_cvt_scale_pk_f16_bf6">;
def int_hcu_cvt_scale_pk_bf16_fp8 : HCUCvtScalePkF32F16FromF8F6Intrinsic<llvm_v2i16_ty, "__builtin_hcu_cvt_scale_pk_bf16_fp8">;
def int_hcu_cvt_scale_pk_bf16_bf8 : HCUCvtScalePkF32F16FromF8F6Intrinsic<llvm_v2i16_ty, "__builtin_hcu_cvt_scale_pk_bf16_bf8">;
def int_hcu_cvt_scale_pk_bf16_fp6 : HCUCvtScalePkF32F16FromF8F6Intrinsic<llvm_v2i16_ty, "__builtin_hcu_cvt_scale_pk_bf16_fp6">;
def int_hcu_cvt_scale_pk_bf16_bf6 : HCUCvtScalePkF32F16FromF8F6Intrinsic<llvm_v2i16_ty, "__builtin_hcu_cvt_scale_pk_bf16_bf6">;

class HCUCvtScalePkF32F16FromF4Intrinsic<LLVMType DstTy, string builtin_name> :
  ClangBuiltin<builtin_name>,
  Intrinsic<[DstTy],       // dst
            [llvm_i32_ty,  // src0
             llvm_i32_ty,  // scale data
             llvm_i32_ty,  // op_sel for src0
             llvm_i1_ty,   // clamp
             llvm_i32_ty], // scale_sel for scale data (select byte [0..3])
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_hcu_cvt_scale_pk_f32_fp4  : HCUCvtScalePkF32F16FromF4Intrinsic<llvm_v2f32_ty, "__builtin_hcu_cvt_scale_pk_f32_fp4">;
def int_hcu_cvt_scale_pk_f16_fp4  : HCUCvtScalePkF32F16FromF4Intrinsic<llvm_v2f16_ty, "__builtin_hcu_cvt_scale_pk_f16_fp4">;
def int_hcu_cvt_scale_pk_bf16_fp4 : HCUCvtScalePkF32F16FromF4Intrinsic<llvm_v2i16_ty, "__builtin_hcu_cvt_scale_pk_bf16_fp4">;

class HCUCvtScaleSrF8FromF32Intrinsic<string builtin_name> :
  ClangBuiltin<builtin_name>,
  Intrinsic<[llvm_i32_ty],  // dst
            [llvm_float_ty, // src0
             llvm_i32_ty,   // seed data
             llvm_i32_ty,   // scale data
             llvm_i32_ty,   // op_sel for dst (select byte [0..3])
             llvm_i32_ty],  // scale_sel for scale data (select byte [0..3])
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_hcu_cvt_scale_sr_fp8_f32  : HCUCvtScaleSrF8FromF32Intrinsic<"__builtin_hcu_cvt_scale_sr_fp8_f32">;
def int_hcu_cvt_scale_sr_bf8_f32  : HCUCvtScaleSrF8FromF32Intrinsic<"__builtin_hcu_cvt_scale_sr_bf8_f32">;

class HCUCvtScaleSrF8FromF16Intrinsic<string builtin_name> :
  ClangBuiltin<builtin_name>,
  Intrinsic<[llvm_i32_ty], // dst
            [llvm_i32_ty,  // src0
             llvm_i32_ty,  // seed data
             llvm_i32_ty,  // scale data
             llvm_i32_ty,  // op_sel for dst (select byte [0..3])
             llvm_i1_ty,   // op_sel for src0 (0:selects 2 low bytes, 1:selects 2 high bytes)
             llvm_i32_ty], // scale_sel for scale data (select byte [0..3])
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_hcu_cvt_scale_sr_fp8_f16  : HCUCvtScaleSrF8FromF16Intrinsic<"__builtin_hcu_cvt_scale_sr_fp8_f16">;
def int_hcu_cvt_scale_sr_bf8_f16  : HCUCvtScaleSrF8FromF16Intrinsic<"__builtin_hcu_cvt_scale_sr_bf8_f16">;
def int_hcu_cvt_scale_sr_fp8_bf16 : HCUCvtScaleSrF8FromF16Intrinsic<"__builtin_hcu_cvt_scale_sr_fp8_bf16">;
def int_hcu_cvt_scale_sr_bf8_bf16 : HCUCvtScaleSrF8FromF16Intrinsic<"__builtin_hcu_cvt_scale_sr_bf8_bf16">;

class HCUCvtScaleSrPkF6F4FromF32F16Intrinsic<LLVMType Src0Ty, LLVMType Src1Ty, string builtin_name> :
  ClangBuiltin<builtin_name>,
  Intrinsic<[llvm_i32_ty], // dst
            [Src0Ty,       // src0
             Src1Ty,       // seed data
             llvm_i32_ty,  // scale data
             llvm_i32_ty,  // op_sel for dst
             llvm_i32_ty], // scale_sel for scale data (select byte [0..3])
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_hcu_cvt_scale_sr_pk_fp6_f32  : HCUCvtScaleSrPkF6F4FromF32F16Intrinsic<llvm_v2f32_ty, llvm_v2i32_ty, "__builtin_hcu_cvt_scale_sr_pk_fp6_f32">;
def int_hcu_cvt_scale_sr_pk_bf6_f32  : HCUCvtScaleSrPkF6F4FromF32F16Intrinsic<llvm_v2f32_ty, llvm_v2i32_ty, "__builtin_hcu_cvt_scale_sr_pk_bf6_f32">;
def int_hcu_cvt_scale_sr_pk_fp4_f32  : HCUCvtScaleSrPkF6F4FromF32F16Intrinsic<llvm_v2f32_ty, llvm_v2i32_ty, "__builtin_hcu_cvt_scale_sr_pk_fp4_f32">;
def int_hcu_cvt_scale_sr_pk_fp6_f16  : HCUCvtScaleSrPkF6F4FromF32F16Intrinsic<llvm_v2f16_ty, llvm_i32_ty,   "__builtin_hcu_cvt_scale_sr_pk_fp6_f16">;
def int_hcu_cvt_scale_sr_pk_bf6_f16  : HCUCvtScaleSrPkF6F4FromF32F16Intrinsic<llvm_v2f16_ty, llvm_i32_ty,   "__builtin_hcu_cvt_scale_sr_pk_bf6_f16">;
def int_hcu_cvt_scale_sr_pk_fp4_f16  : HCUCvtScaleSrPkF6F4FromF32F16Intrinsic<llvm_v2f16_ty, llvm_i32_ty,   "__builtin_hcu_cvt_scale_sr_pk_fp4_f16">;
def int_hcu_cvt_scale_sr_pk_fp6_bf16 : HCUCvtScaleSrPkF6F4FromF32F16Intrinsic<llvm_v2i16_ty, llvm_i32_ty,   "__builtin_hcu_cvt_scale_sr_pk_fp6_bf16">;
def int_hcu_cvt_scale_sr_pk_bf6_bf16 : HCUCvtScaleSrPkF6F4FromF32F16Intrinsic<llvm_v2i16_ty, llvm_i32_ty,   "__builtin_hcu_cvt_scale_sr_pk_bf6_bf16">;
def int_hcu_cvt_scale_sr_pk_fp4_bf16 : HCUCvtScaleSrPkF6F4FromF32F16Intrinsic<llvm_v2i16_ty, llvm_i32_ty,   "__builtin_hcu_cvt_scale_sr_pk_fp4_bf16">;

class HCUCvtPkE8m0FromF32Intrinsic<string builtin_name> :
  ClangBuiltin<builtin_name>,
  Intrinsic<[llvm_i32_ty],  // dst
            [llvm_float_ty, // src0
             llvm_float_ty, // src1
             llvm_i32_ty,   // src2: other bits, dst[x] = src2.b32[x]
             llvm_i1_ty],   // op_sel for dst (0:selects 2 low bytes, 1:selects 2 high bytes)
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<3>>]>;

def int_hcu_cvt_pkrtz_e8m0_f32 : HCUCvtPkE8m0FromF32Intrinsic<"__builtin_hcu_cvt_pkrtz_e8m0_f32">;
def int_hcu_cvt_pkrp_e8m0_f32  : HCUCvtPkE8m0FromF32Intrinsic<"__builtin_hcu_cvt_pkrp_e8m0_f32">;


class HCUPkBF16Intrinsic<string builtin_name>:
  ClangBuiltin<builtin_name>,
  Intrinsic<[llvm_v2i16_ty],
            [llvm_v2i16_ty,
             llvm_v2i16_ty,
             llvm_i32_ty,  // op_sel: default value is 0 ([0,0])
             llvm_i32_ty], // op_sel_hi: default value is 3 ([1,1])
            [IntrNoMem, IntrWillReturn, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_hcu_pk_min_bf16 : HCUPkBF16Intrinsic<"__builtin_hcu_pk_min_bf16">;
def int_hcu_pk_max_bf16 : HCUPkBF16Intrinsic<"__builtin_hcu_pk_max_bf16">;

def int_hcu_dot2_f32_bf16 :
  ClangBuiltin<"__builtin_hcu_dot2_f32_bf16">,
  Intrinsic<[llvm_float_ty],
            [llvm_v2i16_ty,
             llvm_v2i16_ty,
             llvm_float_ty,
             llvm_i32_ty,  // op_sel: default value is 0 ([0,0])
             llvm_i32_ty], // op_sel_hi: default value is 3 ([1,1])
            [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_hcu_get_wave_id : Intrinsic<
  [llvm_i32_ty],
  [llvm_vararg_ty],
  [IntrNoMem]>;

// llvm.hcu.cvt.f32.bf16.dpp float vdst, short src0, imm dpp_ctrl, imm row_mask, imm bank_mask, imm bound_ctrl
def int_hcu_cvt_f32_bf16_dpp : ClangBuiltin<"__builtin_hcu_cvt_f32_bf16_dpp">,
  DefaultAttrsIntrinsic<[llvm_float_ty],
            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty],
            [IntrNoMem, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

// llvm.hcu.cvt.f32.bf8.dpp float vdst, int src0, imm dpp_ctrl, imm row_mask, imm bank_mask, imm bound_ctrl
def int_hcu_cvt_f32_bf8_dpp : ClangBuiltin<"__builtin_hcu_cvt_f32_bf8_dpp">,
  DefaultAttrsIntrinsic<[llvm_float_ty],
            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty],
            [IntrNoMem, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

// llvm.hcu.cvt.f32.fp8.dpp float vdst, int src0, imm dpp_ctrl, imm row_mask, imm bank_mask, imm bound_ctrl
def int_hcu_cvt_f32_fp8_dpp : ClangBuiltin<"__builtin_hcu_cvt_f32_fp8_dpp">,
  DefaultAttrsIntrinsic<[llvm_float_ty],
            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty],
            [IntrNoMem, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

// llvm.hcu.cvt.rtne.tf32.f32.dpp int vdst, imm src_modifier, float src0, imm dpp_ctrl, imm row_mask, imm bank_mask, imm bound_ctrl
def int_hcu_cvt_rtne_tf32_f32_dpp : ClangBuiltin<"__builtin_hcu_cvt_rtne_tf32_f32_dpp">,
  DefaultAttrsIntrinsic<[llvm_i32_ty],
            [llvm_i32_ty, llvm_float_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty],
            [IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

// llvm.hcu.cvt.rna.tf32.f32.dpp int vdst, imm src_modifier, float src0, imm dpp_ctrl, imm row_mask, imm bank_mask, imm bound_ctrl
def int_hcu_cvt_rna_tf32_f32_dpp : ClangBuiltin<"__builtin_hcu_cvt_rna_tf32_f32_dpp">,
  DefaultAttrsIntrinsic<[llvm_i32_ty],
            [llvm_i32_ty, llvm_float_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty],
            [IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

// llvm.hcu.cvt.rtz.tf32.f32.dpp int vdst, imm src_modifier, float src0, imm dpp_ctrl, imm row_mask, imm bank_mask, imm bound_ctrl
def int_hcu_cvt_rtz_tf32_f32_dpp : ClangBuiltin<"__builtin_hcu_cvt_rtz_tf32_f32_dpp">,
  DefaultAttrsIntrinsic<[llvm_i32_ty],
            [llvm_i32_ty, llvm_float_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty],
            [IntrNoMem, ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

// llvm.hcu.cvt.off.fp32.i4 int srcA
def int_hcu_cvt_off_f32_i4: ClangBuiltin<"__builtin_hcu_cvt_off_f32_i4">,
  DefaultAttrsIntrinsic<[llvm_float_ty],
            [llvm_i32_ty],
            [IntrNoMem, IntrSpeculatable]>;

def int_hcu_tensor_load_1 : Intrinsic<
 [],                                  // no return (void)
 [llvm_v4i32_ty,                      // ssamp S# (SGPR)
  llvm_v4i32_ty,                      // rsrc(SGPR)
  llvm_anyptr_ty,                     // soffset
  llvm_i32_ty,                        // offset0
  llvm_i32_ty,                        // offset1
  llvm_i32_ty,                        // sample_num
  llvm_i32_ty,                        // sample_idx
  llvm_i1_ty,                         // glc
  llvm_i1_ty],                        // slc
 [IntrWillReturn, IntrWriteMem, IntrConvergent, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>,
     ImmArg<ArgIndex<7>>, ImmArg<ArgIndex<8>>]>;

// GFX946 packed buffer FP atomics (v2f16 / v2bf16-as-v2i16).
class HCUBufferAtomicPKFP<LLVMType data_ty> : Intrinsic <
  [data_ty],
  [data_ty, llvm_v4i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty],
  [ImmArg<ArgIndex<4>>, IntrWillReturn, IntrNoCallback, IntrNoFree],
            "", [SDNPMemOperand]>,
  AMDGPURsrcIntrinsic<1, 0>;

def int_hcu_buffer_atomic_pk_add_v2f16 :
  HCUBufferAtomicPKFP<llvm_v2f16_ty>;
def int_hcu_buffer_atomic_pk_add_v2bf16 :
  HCUBufferAtomicPKFP<llvm_v2bf16_ty>;
def int_hcu_buffer_atomic_pk_min_v2f16 :
  HCUBufferAtomicPKFP<llvm_v2f16_ty>;
def int_hcu_buffer_atomic_pk_min_v2bf16 :
  HCUBufferAtomicPKFP<llvm_v2bf16_ty>;
def int_hcu_buffer_atomic_pk_max_v2f16 :
  HCUBufferAtomicPKFP<llvm_v2f16_ty>;
def int_hcu_buffer_atomic_pk_max_v2bf16 :
  HCUBufferAtomicPKFP<llvm_v2bf16_ty>;

// GFX946 buffer atomic cmpswap (distinct from amdgcn i32 cmpswap).
def int_hcu_buffer_atomic_cmpswap_f32 :
  Intrinsic<[llvm_v2f32_ty],
            [llvm_v2f32_ty, llvm_v2f32_ty, llvm_v4i32_ty, llvm_i32_ty,
             llvm_i32_ty, llvm_i1_ty],
            [ImmArg<ArgIndex<5>>, IntrWillReturn, IntrNoCallback, IntrNoFree],
            "", [SDNPMemOperand]>,
  AMDGPURsrcIntrinsic<2, 0>;

def int_hcu_buffer_atomic_cmpswap_f64 :
  Intrinsic<[llvm_v2f64_ty],
            [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v4i32_ty, llvm_i32_ty,
             llvm_i32_ty, llvm_i1_ty],
            [ImmArg<ArgIndex<5>>, IntrWillReturn, IntrNoCallback, IntrNoFree],
            "", [SDNPMemOperand]>,
  AMDGPURsrcIntrinsic<2, 0>;

def int_hcu_buffer_atomic_pk_cmpswap_v2f16 :
  Intrinsic<[llvm_v4f16_ty],
            [llvm_v4f16_ty, llvm_v4f16_ty, llvm_v4i32_ty, llvm_i32_ty,
             llvm_i32_ty, llvm_i1_ty],
            [ImmArg<ArgIndex<5>>, IntrWillReturn, IntrNoCallback, IntrNoFree],
            "", [SDNPMemOperand]>,
  AMDGPURsrcIntrinsic<2, 0>;

def int_hcu_buffer_atomic_pk_cmpswap_v2bf16 :
  Intrinsic<[llvm_v4bf16_ty],
            [llvm_v4bf16_ty, llvm_v4bf16_ty, llvm_v4i32_ty, llvm_i32_ty,
             llvm_i32_ty, llvm_i1_ty],
            [ImmArg<ArgIndex<5>>, IntrWillReturn, IntrNoCallback, IntrNoFree],
            "", [SDNPMemOperand]>,
  AMDGPURsrcIntrinsic<2, 0>;

def int_hcu_tensor_load_2 : Intrinsic<
 [],                                  // no return (void)
 [llvm_v4i32_ty,                      // ssamp S# (SGPR)
  llvm_v4i32_ty,                      // rsrc(SGPR)
  llvm_anyptr_ty,                     // soffset
  llvm_i32_ty,                        // offset0
  llvm_i32_ty,                        // offset1
  llvm_i32_ty,                        // sample_num
  llvm_i32_ty,                        // sample_idx
  llvm_i1_ty,                         // glc
  llvm_i1_ty,                         // slc
  llvm_i1_ty],                        // bps

 [IntrWillReturn, IntrWriteMem, IntrConvergent, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>,
     ImmArg<ArgIndex<7>>, ImmArg<ArgIndex<8>>,ImmArg<ArgIndex<9>>]>;

// llvm.hcu.wdra.init(i16 branch1, i16 branch2, i16 branch3, i16 branch4)
// Declares the per-branch VGPR sizes for WDRA, ordered by ascending wave id.
// Each branch covers 4 consecutive waves; at most 16 waves => 4 branches:
//   branch1 -> waves 0-3,  branch2 -> waves 4-7,
//   branch3 -> waves 8-11, branch4 -> waves 12-15.
// Unused branches pass 0. On hardware the HCUInsertWDRAInit pass consumes
// this to emit the kernel-entry wdra prologue (s_trap 0xff + s_set_vgpr_size
// per branch); on the model (PMD) it lowers to a comment and VGPR allocation
// is resolved from HSACO metadata.
def int_hcu_wdra_init : Intrinsic<
  [],
  [llvm_i16_ty, llvm_i16_ty, llvm_i16_ty, llvm_i16_ty],
  [IntrNoMem, IntrHasSideEffects, IntrConvergent,
   ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]
>;
} // TargetPrefix = "hcu"