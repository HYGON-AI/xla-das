// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd., 2026.

//===- ROCDLHCUOps.td - Defines HCU ROCDL Ops -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// ROCDL op definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// HCU Mmac intrinsics
class HCU_Mfma_IntrOp<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase<ROCDL_Dialect, mnemonic,
                  "hcu_" # !subst(".","_", mnemonic),
                  [], [], traits, 1>,
  Arguments<(ins Variadic<LLVM_Type>:$args)> {
  let assemblyFormat =
    "$args attr-dict `:` functional-type($args, $res)";
}

def ROCDL_mmac_f32_16x16x8tf32     : HCU_Mfma_IntrOp<"mmac.f32.16x16x8.tf32">;
def ROCDL_mmac_f32_16x16x4f32       : HCU_Mfma_IntrOp<"mmac.16x16x4.f32">;
def ROCDL_mmac_f32_16x16x8f32       : HCU_Mfma_IntrOp<"mmac.16x16x8.f32">;
def ROCDL_mmac_f32_16x16x16f16      : HCU_Mfma_IntrOp<"mmac.f32.16x16x16.f16">;
def ROCDL_mmac_f32_16x16x16bf16     : HCU_Mfma_IntrOp<"mmac.f32.16x16x16.bf16">;
def ROCDL_mmac_i32_16x16x32i8       : HCU_Mfma_IntrOp<"mmac.i32.16x16x32.i8">;
def ROCDL_mmac_i32_16x16x32u8       : HCU_Mfma_IntrOp<"mmac.i32.16x16x32.u8">;
def ROCDL_mmac_f32_16x16x32_bf8_bf8 : HCU_Mfma_IntrOp<"mmac.f32.16x16x32.bf8.bf8">;
def ROCDL_mmac_f32_16x16x32_bf8_fp8 : HCU_Mfma_IntrOp<"mmac.f32.16x16x32.bf8.fp8">;
def ROCDL_mmac_f32_16x16x32_fp8_bf8 : HCU_Mfma_IntrOp<"mmac.f32.16x16x32.fp8.bf8">;
def ROCDL_mmac_f32_16x16x32_fp8_fp8 : HCU_Mfma_IntrOp<"mmac.f32.16x16x32.fp8.fp8">;

def ROCDL_mmac_f32_16x16x64_fp4     : HCU_Mfma_IntrOp<"mmac.f32.16x16x64.fp4">;
def ROCDL_mmac_f16_16x16x64_fp4     : HCU_Mfma_IntrOp<"mmac.f16.16x16x64.fp4">;
def ROCDL_mmac_bf16_16x16x64_fp4    : HCU_Mfma_IntrOp<"mmac.bf16.16x16x64.fp4">;

def ROCDL_mmac_f32_16x16x32_f8f6f4  : HCU_Mfma_IntrOp<"mmac.f32.16x16x32.f8f6f4">;
def ROCDL_mmac_f16_16x16x32_f8f6f4  : HCU_Mfma_IntrOp<"mmac.f16.16x16x32.f8f6f4">;
def ROCDL_mmac_bf16_16x16x32_f8f6f4 : HCU_Mfma_IntrOp<"mmac.bf16.16x16x32.f8f6f4">;

def ROCDL_mmac_scale_f32_16x16x32_f8f6f4  : HCU_Mfma_IntrOp<"mmac.scale.f32.16x16x32.f8f6f4">;
def ROCDL_mmac_scale_f32_16x16x64_fp4     : HCU_Mfma_IntrOp<"mmac.scale.f32.16x16x64.fp4">;

//===----------------------------------------------------------------------===//
// HCU Matrix Load LDS intrinsics
// note: thie rsrc is very similar to the rsrc with buffer load, but in backend, due to
// backend not define same as buffer load, so we can't use the same class as buffer load.
// reference: http://172.20.48.11/display/DW/User+Guide+for+Built-in+Function
class HCU_MatrixLoadLDS_IntrOp<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase<ROCDL_Dialect, mnemonic,
                  "hcu_" # !subst(".","_", mnemonic),
                  [], [1], traits, 0, 0, 0, 0, 0, 0, [2, 3, 4, 5, 6, 7], ["offset", "t", "r", "glc", "slc", "bps"]>,
  Arguments<(ins LLVM_Type:$rsrc,
                 LLVM_AnyPointer:$soffset,  // soffset (SGPR/imm, excluded from bounds checking and swizzling). smem address.
                 I32Attr:$offset,           // imm (SGPR/imm, excluded from bounds checking and swizzling), the unit is element.
                 I1Attr:$t,                 // imm, 0: column major, 1: row major
                 I1Attr:$r,                 // imm, offset is in  0: column direction, 1: row direction.
                 I1Attr:$glc,               // imm
                 I1Attr:$slc,               // imm
                 I1Attr:$bps)> {            // imm
  let assemblyFormat = "$rsrc `,` $soffset `[` $offset `,` $t `,` $r `,` $glc `,` $slc `,` $bps `]` attr-dict `:` type($rsrc) `,` type($soffset)";
}

def ROCDL_hcu_matrix_load_64X16_b8   : HCU_MatrixLoadLDS_IntrOp<"matrix.load.64x16.b8">;
def ROCDL_hcu_matrix_load_128X16_b8  : HCU_MatrixLoadLDS_IntrOp<"matrix.load.128x16.b8">;
def ROCDL_hcu_matrix_load_64X32_b8   : HCU_MatrixLoadLDS_IntrOp<"matrix.load.64x32.b8">;
def ROCDL_hcu_matrix_load_32X16_b16  : HCU_MatrixLoadLDS_IntrOp<"matrix.load.32x16.b16">;
def ROCDL_hcu_matrix_load_64X16_b16  : HCU_MatrixLoadLDS_IntrOp<"matrix.load.64x16.b16">;
def ROCDL_hcu_matrix_load_32X32_b16  : HCU_MatrixLoadLDS_IntrOp<"matrix.load.32x32.b16">;

def ROCDL_hcu_matrix_load_16X16_b32 : HCU_MatrixLoadLDS_IntrOp<"matrix.load.16x16.b32">;
def ROCDL_hcu_matrix_load_64X16_b6 : HCU_MatrixLoadLDS_IntrOp<"matrix.load.64x16.b6">;
def ROCDL_hcu_matrix_load_64X16_u4 : HCU_MatrixLoadLDS_IntrOp<"matrix.load.64x16.u4">;
def ROCDL_hcu_matrix_load_64X16_i4 : HCU_MatrixLoadLDS_IntrOp<"matrix.load.64x16.i4">;
def ROCDL_hcu_matrix_load_64X16_fp4 : HCU_MatrixLoadLDS_IntrOp<"matrix.load.64x16.fp4">;
def ROCDL_hcu_matrix_load_128X16_b4 : HCU_MatrixLoadLDS_IntrOp<"matrix.load.128x16.b4">;
def ROCDL_hcu_matrix_load_256X16_b4 : HCU_MatrixLoadLDS_IntrOp<"matrix.load.256x16.b4">;


// HCU DS Read Matrix Format intrinsics
class HCU_DsReadMatrixFormat_IntrOp<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase<ROCDL_Dialect, mnemonic,
                  "hcu_" # !subst(".","_", mnemonic),
                  [0], [0], traits, 1, 0, 0, 0, 0, 0, [1, 2, 3, 4, 5], ["offset", "element", "row", "col", "alt"]>,
  Arguments<(ins LLVM_AnyPointer:$soffset,
                 I32Attr:$offset,         // imm, byte offset, smem address = soffset + offset
                 I8Attr:$element,         // imm, 0: 4bit, 1: 8bit, 2:16bit, 3:32bit element
                 I8Attr:$row,             // imm, matrix row size, 0: 8 elem, 1: 16 elem, 2: 32 elem, 3: 64 elem, 4: 128 elem
                 I8Attr:$col,             // imm, matrix col size, 0: 8 elem, 1: 16 elem, 2: 32 elem, 3: 64 elem, 4: 128 elem
                 I8Attr:$alt)> {          // imm, 0: non-interleaved, 1: 2interleaved, 2: 4interleaved, 3: 8interleaved
  let results = (outs LLVM_AnyVector:$res);
  let assemblyFormat = "$soffset `[` $offset `,` $element `,` $row `,` $col `,` $alt `]` attr-dict `:` type($soffset) `->` type($res)";
}

def ROCDL_hcu_ds_read_matrix_format       : HCU_DsReadMatrixFormat_IntrOp<"ds.read.matrix.format">;
def ROCDL_hcu_ds_read_matrix_trans_format : HCU_DsReadMatrixFormat_IntrOp<"ds.read.matrix.trans.format">;
def ROCDL_hcu_ds_read_matrix_padbyte      : HCU_DsReadMatrixFormat_IntrOp<"ds.read.matrix.padbyte">;
def ROCDL_hcu_ds_read_matrix_trans_padbyte: HCU_DsReadMatrixFormat_IntrOp<"ds.read.matrix.trans.padbyte">;



class ROCDL_HCU_IntrOp<string mnemonic, list<int> overloadedResults,
  list<int> overloadedOperands, list<Trait> traits, int numResults,
  int requiresAccessGroup = 0, int requiresAliasAnalysis = 0, list<int> immArgPositions = [],
  list<string> immArgAttrNames = []> :
  LLVM_IntrOpBase<ROCDL_Dialect,  "hcu."#mnemonic,
    "hcu_" # !subst(".", "_", mnemonic), overloadedResults,
    overloadedOperands, traits, numResults, requiresAccessGroup,
    requiresAliasAnalysis, 0, 0, 0, immArgPositions, immArgAttrNames>;


//===----------------------------------------------------------------------===//
// CVT definitions

// Available since gfx938
// TODO: add more
def ROCDL_HCUCvtF32Bf8Op :
    ROCDL_HCU_IntrOp<"cvt.f32.bf8", [], [], [Pure], 1>,
    Arguments<(ins I32:$srcA, I32:$byteSel)> {
  let summary = "Convert bf8 to f32";
  let description = [{
    Convert 8-bit bf8 value from the `byteSel`th bit of `srcA` to fp32.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `[` $byteSel `]` `:` type($res)
  }];
}

def ROCDL_HCUCvtF32Fp8Op :
    ROCDL_HCU_IntrOp<"cvt.f32.fp8", [], [], [Pure], 1>,
    Arguments<(ins I32:$srcA, I32:$byteSel)> {
  let summary = "Convert fp8 to f32";
  let description = [{
    Convert 8-bit fp8 value from the `byteSel`th bit of `srcA` to fp32.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `[` $byteSel `]` `:` type($res)
  }];
}

def ROCDL_HCUCvtPkF32Fp8Op :
    ROCDL_HCU_IntrOp<"cvt.pk.f32.fp8", [], [], [Pure], 1>,
    Arguments<(ins I32:$src, I1:$wordSel)> {
  let summary = "Convert packed fp8 to packed f32";
  let description = [{
    Convert `src` based on $wordSel to packed fp32.
  }];
  let assemblyFormat = [{
    attr-dict $src `[` $wordSel `]` `:` type($res)
  }];
}

def ROCDL_HCUCvtPkF32Bf8Op :
    ROCDL_HCU_IntrOp<"cvt.pk.f32.bf8", [], [], [Pure], 1>,
    Arguments<(ins I32:$src, I1:$wordSel)> {
  let summary = "Convert packed bf8 to packed f32";
  let description = [{
    Convert `src` based on $wordSel to packed fp32,
  }];
  let assemblyFormat = [{
    attr-dict $src `[` $wordSel `]` `:` type($res)
  }];
}

def ROCDL_HCUCvtPkBf8F32Op :
    ROCDL_HCU_IntrOp<"cvt.pk.bf8.f32", [], [], [Pure], 1>,
    Arguments<(ins F32:$srcA, F32:$srcB, I32:$old, I1:$wordSel)> {
  let summary = "Convert two f32's to bf8";
  let description = [{
    Convert `srcA` and `srcB` to bf8 and store into the low/high word of
    `old`, preserving the other word.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `,` $srcB `->` $old `[` $wordSel `]` `:` type($res)
  }];
}

def ROCDL_HCUCvtPkFp8F32Op :
    ROCDL_HCU_IntrOp<"cvt.pk.fp8.f32", [], [], [Pure], 1>,
    Arguments<(ins F32:$srcA, F32:$srcB, I32:$old, I1:$wordSel)> {
  let summary = "Convert two f32's to fp8";
  let description = [{
    Convert `srcA` and `srcB` to fp8 and store into the low/high word of
    `old`, preserving the other word.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `,` $srcB `->` $old `[` $wordSel `]` `:` type($res)
  }];
}

def ROCDL_HCUCvtSrBf8F32Op :
    ROCDL_HCU_IntrOp<"cvt.sr.bf8.f32", [], [], [Pure], 1>,
    Arguments<(ins F32:$srcA, I32:$srcB, I32:$old, I32:$byteSel)> {
  let summary = "Convert f32 to bf8, stochiastic rounding";
  let description = [{
    Convert `srcA` to bf8, adding the rounding factor from `srcB`,
    and store into the `byteSel`th byte of `old`, preserving the others.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `,` $srcB `->` $old `[` $byteSel `]` `:` type($res)
  }];
}

def ROCDL_HCUCvtSrFp8F32Op :
    ROCDL_HCU_IntrOp<"cvt.sr.fp8.f32", [], [], [Pure], 1>,
    Arguments<(ins F32:$srcA, I32:$srcB, I32:$old, I32:$byteSel)> {
  let summary = "Convert f32 to fp8, stochiastic rounding";
  let description = [{
    Convert `srcA` to fp8, adding the rounding factor from `srcB`,
    and store into the `byteSel`th byte of `old`, preserving the others.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `,` $srcB `->` $old `[` $byteSel `]` `:` type($res)
  }];
}

def ROCDL_HCUCvtF32Bf16Op :
    ROCDL_HCU_IntrOp<"cvt.f32.bf16", [], [], [Pure], 1>,
    Arguments<(ins I32:$src, I1:$clamp, I32:$omod, I1:$wordSel)> {
  let summary = "Convert bf16 to f32";
  let description = [{
    Convert bf16 value from `src` to f32. The `wordSel` selects which half
    of the i32 input to use.
  }];
  let assemblyFormat = [{
    attr-dict $src `,` $clamp `,` $omod `[` $wordSel `]` `:` type($res)
  }];
}

def ROCDL_HCUCvtBf16F32Op :
    ROCDL_HCU_IntrOp<"cvt.bf16.f32", [], [], [Pure], 1>,
    Arguments<(ins F32:$src, I1:$clamp, I1:$wordSel)> {
  let summary = "Convert f32 to bf16";
  let description = [{
    Convert f32 value to bf16. The `wordSel` selects which half
    of the f32 output to use.
  }];
  let assemblyFormat = [{
    attr-dict $src `,` $clamp `[` $wordSel `]` `:` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// ebarrier definitions
// Available on gfx92a, gfx946, gfx948

def ROCDL_HCUEbarrierArriveOp :
    ROCDL_HCU_IntrOp<"s.ebarrier.arrive", [], [], [], 0>,
    Arguments<(ins I32:$barId, I32Attr:$wvCnt)> {
  let summary = "Enhanced barrier arrive";
  let description = [{
    Perform an arrive action on the ebarrier object with barId.
    This action is un-blockable.
  }];
  let builders = [
    OpBuilder<(ins "Value":$barId, "int":$wvCnt), [{
      build($_builder, $_state, barId, $_builder.getI32IntegerAttr(wvCnt));
    }]>,
    OpBuilder<(ins "int":$barId, "int":$wvCnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      build($_builder, $_state, barIdValue, $_builder.getI32IntegerAttr(wvCnt));
    }]>
  ];
  let llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_ebarrier_arrive, {$barId, builder.getInt32($wvCnt)});
  }];
  let assemblyFormat = [{
    attr-dict $barId `,` $wvCnt
  }];
}

def ROCDL_HCUEbarrierSyncOp :
    ROCDL_HCU_IntrOp<"s.ebarrier.sync", [], [], [], 0>,
    Arguments<(ins I32:$barId, I32Attr:$wvCnt)> {
  let summary = "Enhanced barrier sync";
  let description = [{
    Perform a sync action on the ebarrier object with barId.
    This action is blockable.
  }];
  let builders = [
    OpBuilder<(ins "Value":$barId, "int":$wvCnt), [{
      build($_builder, $_state, barId, $_builder.getI32IntegerAttr(wvCnt));
    }]>,
    OpBuilder<(ins "int":$barId, "int":$wvCnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      build($_builder, $_state, barIdValue, $_builder.getI32IntegerAttr(wvCnt));
    }]>
  ];
  let llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_ebarrier_sync, {$barId, builder.getInt32($wvCnt)});
  }];
  let assemblyFormat = [{
    attr-dict $barId `,` $wvCnt
  }];
}

def ROCDL_HCUEbarrierAndOp :
    ROCDL_HCU_IntrOp<"s.ebarrier.and", [], [], [], 1>,
    Arguments<(ins I32:$barId, I32Attr:$wvCnt)> {
  let summary = "Enhanced barrier and";
  let description = [{
    Perform bitwise "AND" operation on the value (one bit per thread) of all threads participating this barrier.
    This op reads implicitly VCC_HI/LO (2 sgprs) for values to be reduced.
    The result is a boolean value that will be written to the dest sgpr.
    This action is blockable.
  }];
  let results = (outs I32:$res);
  let builders = [
    OpBuilder<(ins "Type":$res, "Value":$barId, "int":$wvCnt), [{
      build($_builder, $_state, res, barId, $_builder.getI32IntegerAttr(wvCnt));
    }]>,
    OpBuilder<(ins "Value":$barId, "int":$wvCnt), [{
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barId, $_builder.getI32IntegerAttr(wvCnt));
    }]>,
    OpBuilder<(ins "Type":$res, "int":$barId, "int":$wvCnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      build($_builder, $_state, res, barIdValue, $_builder.getI32IntegerAttr(wvCnt));
    }]>,
    OpBuilder<(ins "int":$barId, "int":$wvCnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barIdValue, $_builder.getI32IntegerAttr(wvCnt));
    }]>
  ];
  let llvmBuilder = [{
    $res = createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_ebarrier_and, {$barId, builder.getInt32($wvCnt)});
  }];
  let assemblyFormat = [{
    attr-dict type($res) `,` $barId `,` $wvCnt
  }];
}

def ROCDL_HCUEbarrierOrOp :
    ROCDL_HCU_IntrOp<"s.ebarrier.or", [], [], [], 1>,
    Arguments<(ins I32:$barId, I32Attr:$wvCnt)> {
  let summary = "Enhanced barrier or";
  let description = [{
    Perform bitwise "or" operation on the value (one bit per thread) of all threads participating this barrier.
    This op reads implicitly VCC_HI/LO (2 sgprs) for values to be reduced.
    The result is a boolean value that will be written to the dest sgpr.
    This action is blockable.
  }];
  let results = (outs I32:$res);
  let builders = [
    OpBuilder<(ins "Type":$res, "Value":$barId, "int":$wvCnt), [{
      build($_builder, $_state, res, barId, $_builder.getI32IntegerAttr(wvCnt));
    }]>,
    OpBuilder<(ins "Value":$barId, "int":$wvCnt), [{
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barId, $_builder.getI32IntegerAttr(wvCnt));
    }]>,
    OpBuilder<(ins "Type":$res, "int":$barId, "int":$wvCnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      build($_builder, $_state, res, barIdValue, $_builder.getI32IntegerAttr(wvCnt));
    }]>,
    OpBuilder<(ins "int":$barId, "int":$wvCnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barIdValue, $_builder.getI32IntegerAttr(wvCnt));
    }]>
  ];
  let llvmBuilder = [{
    $res = createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_ebarrier_or, {$barId, builder.getInt32($wvCnt)});
  }];
  let assemblyFormat = [{
    attr-dict type($res) `,` $barId `,` $wvCnt
  }];
}

def ROCDL_HCUEbarrierPopcOp :
    ROCDL_HCU_IntrOp<"s.ebarrier.popc", [], [], [], 1>,
    Arguments<(ins I32:$barId, I32Attr:$wvCnt)> {
  let summary = "Enhanced barrier popc";
  let description = [{
    Count the number of ones on the value (one bit per thread) of all threads participating this barrier.
    This op reads implicitly VCC_HI/LO (2 sgprs) for values to be reduced.
    The result is a boolean value that will be written to the dest sgpr.
    This action is blockable.
  }];
  let results = (outs I32:$res);
  let builders = [
    OpBuilder<(ins "Type":$res, "Value":$barId, "int":$wvCnt), [{
      build($_builder, $_state, res, barId, $_builder.getI32IntegerAttr(wvCnt));
    }]>,
    OpBuilder<(ins "Value":$barId, "int":$wvCnt), [{
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barId, $_builder.getI32IntegerAttr(wvCnt));
    }]>,
    OpBuilder<(ins "Type":$res, "int":$barId, "int":$wvCnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      build($_builder, $_state, res, barIdValue, $_builder.getI32IntegerAttr(wvCnt));
    }]>,
    OpBuilder<(ins "int":$barId, "int":$wvCnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barIdValue, $_builder.getI32IntegerAttr(wvCnt));
    }]>
  ];
  let llvmBuilder = [{
    $res = createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_ebarrier_popc, {$barId, builder.getInt32($wvCnt)});
  }];
  let assemblyFormat = [{
    attr-dict type($res) `,` $barId `,` $wvCnt
  }];
}

def ROCDL_HCUEbarrierObjectOp :
    ROCDL_HCU_IntrOp<"s.ebarrier.slot.rd", [], [], [], 1>,
    Arguments<(ins I32:$barId)> {
  let summary = "Read the enhanced barrier slot";
  let description = [{
    Read the value of ebarrier slot to sgpr.
    32-bit dst value:
    {8'h0, 11-bit red_data, 3-bit red_op, 4-bit arrival_count,
     5-bit expected_arrival_count, 1-bit valid}
  }];
  let results = (outs I32:$res);
  let builders = [
    OpBuilder<(ins "Value":$barId), [{
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barId);
    }]>,
    OpBuilder<(ins "int":$barId), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barIdValue);
    }]>
  ];
  let llvmBuilder = [{
    $res = createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_ebarrier_slot_rd, {$barId});
  }];
  let assemblyFormat = [{
    attr-dict type($res) `,` $barId
  }];
}

//===----------------------------------------------------------------------===//
// abarrier definitions
// Available on gfx92a, gfx946, gfx948

def ROCDL_HCUAbarrierInitOp :
    ROCDL_HCU_IntrOp<"s.abarrier.init", [], [], [], 0>,
    Arguments<(ins I32:$barId, I32Attr:$cnt)> {
  let summary = "Initialize the asynchronois barrier";
  let description = [{
    Initialize the asynchronous barrier. Set the expected arrival count to be cnt.
    Set the pending arrival count to be cnt. Set the transaction count to be 0.
    Set the phase to be 0. Set the valid bit to be 1.
    This op do not sync the waves. Have to use ebarrier sync after initialization.
  }];
  let builders = [
    OpBuilder<(ins "Value":$barId, "int":$cnt), [{
      build($_builder, $_state, barId, $_builder.getI32IntegerAttr(cnt));
    }]>,
    OpBuilder<(ins "int":$barId, "int":$cnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      build($_builder, $_state, barIdValue, $_builder.getI32IntegerAttr(cnt));
    }]>
  ];
  let llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_abarrier_init, {$barId, builder.getInt32($cnt)});
  }];
  let assemblyFormat = [{
    attr-dict $barId `,` $cnt
  }];
}

def ROCDL_HCUAbarrierInvOp :
    ROCDL_HCU_IntrOp<"s.abarrier.inv", [], [], [], 0>,
    Arguments<(ins I32:$barId)> {
  let summary = "Invalidate the asynchronois barrier";
  let description = [{
    Invalidate the asynchronous barrier. Release the abarrier slot.
    Signal the end of the abarrier object.
  }];
  let builders = [
    OpBuilder<(ins "int":$barId), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      build($_builder, $_state, barIdValue);
    }]>
  ];
  let llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_abarrier_inv, {$barId});
  }];
  let assemblyFormat = [{
    attr-dict $barId
  }];
}

def ROCDL_HCUAbarrierSeqOp :
    ROCDL_HCU_IntrOp<"s.abarrier.seq", [], [], [], 0>,
    Arguments<(ins I32:$barId)> {
  let summary = "Asynchronois barrier sequence";
  let description = [{
    Mark the abarrier would track the vector memory operations afterwards, including:
    1. Between two abarrier_seq instrutions;
    2. Between the abarrier_seq instruction and abarrier_arrive instruction with the same barrier id;
  }];
  let builders = [
    OpBuilder<(ins "int":$barId), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      build($_builder, $_state, barIdValue);
    }]>
  ];
  let llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_abarrier_seq, {$barId});
  }];
  let assemblyFormat = [{
    attr-dict $barId
  }];
}

def ROCDL_HCUAbarrierExpectTxOp :
    ROCDL_HCU_IntrOp<"s.abarrier.expect.tx", [], [], [], 0>,
    Arguments<(ins I32:$barId, I32Attr:$cnt)> {
  let summary = "Asynchronois barrier expect transaction";
  let description = [{
    Add cnt to the data size this barrier have to sync.
    For bypass L1 vector memory instrs, cnt is the bytes number;
    For non-bypass L1 vector memory instrs, cnt is the number of instrs;
  }];
  let builders = [
    OpBuilder<(ins "Value":$barId, "int":$cnt), [{
      build($_builder, $_state, barId, $_builder.getI32IntegerAttr(cnt));
    }]>,
    OpBuilder<(ins "int":$barId, "int":$cnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      build($_builder, $_state, barIdValue, $_builder.getI32IntegerAttr(cnt));
    }]>
  ];
  let llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_abarrier_expect_tx, {$barId, builder.getInt32($cnt)});
  }];
  let assemblyFormat = [{
    attr-dict $barId `,` $cnt
  }];
}

def ROCDL_HCUAbarrierCompleteTxOp :
    ROCDL_HCU_IntrOp<"s.abarrier.complete.tx", [], [], [], 0>,
    Arguments<(ins I32:$barId, I32Attr:$cnt)> {
  let summary = "Asynchronois barrier complete transaction";
  let description = [{
    Decrease cnt to the data size this barrier have to sync.
  }];
  let builders = [
    OpBuilder<(ins "Value":$barId, "int":$cnt), [{
      build($_builder, $_state, barId, $_builder.getI32IntegerAttr(cnt));
    }]>,
    OpBuilder<(ins "int":$barId, "int":$cnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      build($_builder, $_state, barIdValue, $_builder.getI32IntegerAttr(cnt));
    }]>
  ];
  let llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_abarrier_complete_tx, {$barId, builder.getInt32($cnt)});
  }];
  let assemblyFormat = [{
    attr-dict $barId `,` $cnt
  }];
}

def ROCDL_HCUAbarrierArriveOp :
    ROCDL_HCU_IntrOp<"s.abarrier.arrive", [], [], [], 1>,
    Arguments<(ins I32:$barId, I32Attr:$cnt)> {
  let summary = "Asynchronois barrier arrive on";
  let description = [{
    Perform an arrive-on on this abarrier object, pending_arrival_count decreace by cnt.
    Return the state of this abarrier to sgpr/m0.
    Cnt normally is 1, or taken as 1 if cnt is 0 or not provided.
    Mark the tracking of the previous vector memory instrs is ended.
    32-bit state:
    {5'h0, 11-bit pending_arrival_count, 15'h0, 1-bit phase}
  }];
  let results = (outs I32:$res);
  let builders = [
    OpBuilder<(ins "Type":$res, "Value":$barId, "int":$cnt), [{
      build($_builder, $_state, res, barId, $_builder.getI32IntegerAttr(cnt));
    }]>,
    OpBuilder<(ins "Value":$barId, "int":$cnt), [{
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barId, $_builder.getI32IntegerAttr(cnt));
    }]>,
    OpBuilder<(ins "Type":$res, "int":$barId, "int":$cnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      build($_builder, $_state, res, barIdValue, $_builder.getI32IntegerAttr(cnt));
    }]>,
    OpBuilder<(ins "int":$barId, "int":$cnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barIdValue, $_builder.getI32IntegerAttr(cnt));
    }]>
  ];
  let llvmBuilder = [{
    $res = createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_abarrier_arrive, {$barId, builder.getInt32($cnt)});
  }];
  let assemblyFormat = [{
    attr-dict type($res) `,` $barId `,` $cnt
  }];
}

def ROCDL_HCUAbarrierArriveNoCompleteOp :
    ROCDL_HCU_IntrOp<"s.abarrier.arrive.nocomplete", [], [], [], 1>,
    Arguments<(ins I32:$barId, I32Attr:$cnt)> {
  let summary = "Asynchronois barrier arrive no complete";
  let description = [{
    Perform an arrive-on on this abarrier object, pending_arrival_count decreace by cnt.
    Return the state of this abarrier to sgpr/m0.
    Cnt normally is 1, or taken as 1 if cnt is 0 or not provided.
    Explicitly mark that this op does not end the current phase. If it does, hardware would report an error.
    32-bit state:
    {5'h0, 11-bit pending_arrival_count, 15'h0, 1-bit phase}
  }];
  let results = (outs I32:$res);
  let builders = [
    OpBuilder<(ins "Type":$res, "Value":$barId, "int":$cnt), [{
      build($_builder, $_state, res, barId, $_builder.getI32IntegerAttr(cnt));
    }]>,
    OpBuilder<(ins "Value":$barId, "int":$cnt), [{
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barId, $_builder.getI32IntegerAttr(cnt));
    }]>,
    OpBuilder<(ins "Type":$res, "int":$barId, "int":$cnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      build($_builder, $_state, res, barIdValue, $_builder.getI32IntegerAttr(cnt));
    }]>,
    OpBuilder<(ins "int":$barId, "int":$cnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barIdValue, $_builder.getI32IntegerAttr(cnt));
    }]>
  ];
  let llvmBuilder = [{
    $res = createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_abarrier_arrive_nocomplete, {$barId, builder.getInt32($cnt)});
  }];
  let assemblyFormat = [{
    attr-dict type($res) `,` $barId `,` $cnt
  }];
}

def ROCDL_HCUAbarrierArriveDropOp :
    ROCDL_HCU_IntrOp<"s.abarrier.arrive.drop", [], [], [], 1>,
    Arguments<(ins I32:$barId, I32Attr:$cnt)> {
  let summary = "Asynchronois barrier arrive drop";
  let description = [{
    Perform an arrive-on on this abarrier object, pending_arrival_count decreace by cnt.
    Return the state of this abarrier to sgpr/m0.
    Current wave drops from this barrier. The barrier expect_arriva_count decreases by 1.
    New expected_arrival_count takes effect after current phase is ended.
    Cnt normally is 1, or taken as 1 if cnt is 0 or not provided.
    32-bit state:
    {5'h0, 11-bit pending_arrival_count, 15'h0, 1-bit phase}
  }];
  let results = (outs I32:$res);
  let builders = [
    OpBuilder<(ins "Type":$res, "Value":$barId, "int":$cnt), [{
      build($_builder, $_state, res, barId, $_builder.getI32IntegerAttr(cnt));
    }]>,
    OpBuilder<(ins "Value":$barId, "int":$cnt), [{
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barId, $_builder.getI32IntegerAttr(cnt));
    }]>,
    OpBuilder<(ins "Type":$res, "int":$barId, "int":$cnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      build($_builder, $_state, res, barIdValue, $_builder.getI32IntegerAttr(cnt));
    }]>,
    OpBuilder<(ins "int":$barId, "int":$cnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barIdValue, $_builder.getI32IntegerAttr(cnt));
    }]>
  ];
  let llvmBuilder = [{
    $res = createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_abarrier_arrive_drop, {$barId, builder.getInt32($cnt)});
  }];
  let assemblyFormat = [{
    attr-dict type($res) `,` $barId `,` $cnt
  }];
}

def ROCDL_HCUAbarrierArriveDropNoCompleteOp :
    ROCDL_HCU_IntrOp<"s.abarrier.arrive.drop.nocomplete", [], [], [], 1>,
    Arguments<(ins I32:$barId, I32Attr:$cnt)> {
  let summary = "Asynchronois barrier arrive drop no complete";
  let description = [{
    The synthesis of HCUAbarrierArriveDrop and HcuAbarrierArriveNoComplete.
  }];
  let results = (outs I32:$res);
  let builders = [
    OpBuilder<(ins "Type":$res, "Value":$barId, "int":$cnt), [{
      build($_builder, $_state, res, barId, $_builder.getI32IntegerAttr(cnt));
    }]>,
    OpBuilder<(ins "Value":$barId, "int":$cnt), [{
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barId, $_builder.getI32IntegerAttr(cnt));
    }]>,
    OpBuilder<(ins "Type":$res, "int":$barId, "int":$cnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      build($_builder, $_state, res, barIdValue, $_builder.getI32IntegerAttr(cnt));
    }]>,
    OpBuilder<(ins "int":$barId, "int":$cnt), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barIdValue, $_builder.getI32IntegerAttr(cnt));
    }]>
  ];
  let llvmBuilder = [{
    $res = createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_abarrier_arrive_drop_nocomplete, {$barId, builder.getInt32($cnt)});
  }];
  let assemblyFormat = [{
    attr-dict type($res) `,` $barId `,` $cnt
  }];
}

def ROCDL_HCUAbarrierTryWaitOp :
    ROCDL_HCU_IntrOp<"s.abarrier.try.wait", [], [], [], 1>,
    Arguments<(ins I32:$barId, I32:$phase)> {
  let summary = "Asynchronois barrier try wait";
  let description = [{
    Check if the current phase of the abarrier object is ended.
    This op blocks current wave for a default suspendTime (SQ_SYNC1).
    If the phase is ended, write the signal (boolean) to sgpr/m0.
  }];
  let results = (outs I32:$res);
  let builders = [
    OpBuilder<(ins "Value":$barId, "Value":$phase), [{
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barId, phase);
    }]>,
    OpBuilder<(ins "int":$barId, "Value":$phase), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barIdValue, phase);
    }]>
  ];
  let llvmBuilder = [{
    $res = createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_abarrier_try_wait, {$barId, $phase});
  }];
  let assemblyFormat = [{
    attr-dict type($res) `,` $barId `,` $phase
  }];
}

def ROCDL_HCUAbarrierTestWaitOp :
    ROCDL_HCU_IntrOp<"s.abarrier.test.wait", [], [], [], 1>,
    Arguments<(ins I32:$barId, I32:$phase)> {
  let summary = "Asynchronois barrier test wait";
  let description = [{
    Check if the current phase of the abarrier object is ended.
    This op does not block current wave.
    If the phase is ended, write the signal (boolean) to sgpr/m0.
  }];
  let results = (outs I32:$res);
  let builders = [
    OpBuilder<(ins "Value":$barId, "Value":$phase), [{
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barId, phase);
    }]>,
    OpBuilder<(ins "int":$barId, "Value":$phase), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barIdValue, phase);
    }]>
  ];
  let llvmBuilder = [{
    $res = createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_abarrier_test_wait, {$barId, $phase});
  }];
  let assemblyFormat = [{
    attr-dict type($res) `,` $barId `,` $phase
  }];
}

def ROCDL_HCUAbarrierObjectOp :
    ROCDL_HCU_IntrOp<"s.abarrier.slot.rd", [], [], [], 1>,
    Arguments<(ins I32:$barId)> {
  let summary = "Read the asynchronous barrier slot";
  let description = [{
    Read the state value of this abarrier object to sgpr.
  }];
  let results = (outs I32:$res);
  let builders = [
    OpBuilder<(ins "Value":$barId), [{
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barId);
    }]>,
    OpBuilder<(ins "int":$barId), [{
      auto barIdValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), $_builder.getIntegerType(32),
          $_builder.getI32IntegerAttr(barId));
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType, barIdValue);
    }]>
  ];
  let llvmBuilder = [{
    $res = createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_abarrier_slot_rd, {$barId});
  }];
  let assemblyFormat = [{
    attr-dict type($res) `,` $barId
  }];
}

//===----------------------------------------------------------------------===//
// Get wave id
//===----------------------------------------------------------------------===//

def ROCDL_HCUGetWaveIdOp :
    ROCDL_HCU_IntrOp<"get.wave.id", [], [], [], 1>,
    Arguments<(ins)> {
  let summary = "Get the current wave id";
  let description = [{
    Get the current wave id
  }];
  let results = (outs I32:$res);
  let builders = [
    OpBuilder<(ins), [{
      auto resType = IntegerType::get($_builder.getContext(), 32);
      build($_builder, $_state, resType);
    }]>,
  ];
  let llvmBuilder = [{
    $res = createIntrinsicCall(builder, llvm::Intrinsic::hcu_get_wave_id, {});
  }];
  let assemblyFormat = [{
    attr-dict type($res)
  }];
}

//===----------------------------------------------------------------------===//
// Permlane swap (gfx946)
//===----------------------------------------------------------------------===//

// Match ROCDL_Permlane{16,32}SwapOp: ImmArgs at positions 2/3 are attributes,
// not SSA values. Do not use a custom llvmBuilder with Attrs in the Value list.
def ROCDL_HCUPermlane16SwapOp :
    ROCDL_HCU_IntrOp<"permlane16.swap.b32", [], [], [Pure], 1, 0, 0,
                     [2, 3], ["boundCtrl", "fi"]>,
    Arguments<(ins I32:$dstIn, I32:$src, I1Attr:$boundCtrl, I1Attr:$fi)> {
  let summary = "HCU permlane16 swap";
  let description = [{
    Odd rows of the first operand are swapped with even rows of the second
    operand (one row is 16 lanes). Returns {dst, src_out}.
  }];
  let results = (outs ROCDL_ConcretePair<I32, I32>:$res);
  let assemblyFormat = [{
    attr-dict $dstIn `,` $src `,` $boundCtrl `,` $fi `:` `(` type($dstIn) `,` type($src) `)` `->` type($res)
  }];
}

def ROCDL_HCUPermlane32SwapOp :
    ROCDL_HCU_IntrOp<"permlane32.swap.b32", [], [], [Pure], 1, 0, 0,
                     [2, 3], ["boundCtrl", "fi"]>,
    Arguments<(ins I32:$dstIn, I32:$src, I1Attr:$boundCtrl, I1Attr:$fi)> {
  let summary = "HCU permlane32 swap";
  let description = [{
    Rows 2 and 3 of the first operand are swapped with rows 0 and 1 of the
    second operand (one row is 16 lanes). Returns {dst, src_out}.
  }];
  let results = (outs ROCDL_ConcretePair<I32, I32>:$res);
  let assemblyFormat = [{
    attr-dict $dstIn `,` $src `,` $boundCtrl `,` $fi `:` `(` type($dstIn) `,` type($src) `)` `->` type($res)
  }];
}

def ROCDL_HCUSetVgprSizeOp :
    ROCDL_HCU_IntrOp<"s.set.vgpr.size", [], [], [], 0>,
    Arguments<(ins I16:$size)> {
  let summary = "Set the total number of vgpr size for calling branch.";
  let description = [{
    Set number of vgpr.
  }];
  let builders = [
    OpBuilder<(ins "int":$size), [{
      auto sizeType = $_builder.getIntegerType(16);
      auto sizeValue = $_builder.create<LLVM::ConstantOp>(
          $_builder.getUnknownLoc(), sizeType,
          $_builder.getIntegerAttr(sizeType, size));
      build($_builder, $_state, sizeValue);
    }]>
  ];
  let llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::hcu_s_set_vgpr_size, {$size});
  }];
  let assemblyFormat = [{
    attr-dict $size
  }];
}

def ROCDL_HCUWdraInitOp :
    ROCDL_HCU_IntrOp<"wdra.init", [], [], [], 0,
                     /*requiresAccessGroup=*/0, /*requiresAliasAnalysis=*/0,
                     /*immArgPositions=*/[0, 1, 2, 3],
                     /*immArgAttrNames=*/["branch1", "branch2", "branch3", "branch4"]> {
  let summary = "Declare per-branch VGPR sizes for WDRA.";
  let description = [{
    Declare the per-branch VGPR sizes for WDRA, ordered by ascending wave id.
    Each branch covers 4 consecutive waves; at most 16 waves => 4 branches:
      branch1 -> waves 0-3,  branch2 -> waves 4-7,
      branch3 -> waves 8-11, branch4 -> waves 12-15.
    Unused branches pass 0. On hardware the backend HCUInsertWDRAInit pass
    lowers this to the kernel-entry wdra prologue (s_trap 0xff +
    s_set_vgpr_size per branch); on the model (PMD) it lowers to a comment and
    VGPR allocation is resolved from HSACO metadata.
  }];
  let arguments = (ins I16Attr:$branch1, I16Attr:$branch2,
                       I16Attr:$branch3, I16Attr:$branch4);
  let builders = [
    OpBuilder<(ins "int":$branch1, "int":$branch2, "int":$branch3, "int":$branch4),
      [{
        build($_builder, $_state,
              $_builder.getI16IntegerAttr(branch1),
              $_builder.getI16IntegerAttr(branch2),
              $_builder.getI16IntegerAttr(branch3),
              $_builder.getI16IntegerAttr(branch4));
      }]>
  ];
  let assemblyFormat = [{
    attr-dict
  }];
}
