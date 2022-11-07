from __future__ import annotations
# https://arxiv.org/pdf/2112.10752.pdf
# https://github.com/ekagra-ranjan/huggingface-blog/blob/main/stable_diffusion.md
# Blatantly copied from:
# https://github.com/geohot/tinygrad/blob/master/examples/stable_diffusion.py
# https://github.com/geohot/tinygrad/blob/master/LICENSE
import os
import gzip
import argparse
import math
import re
import pickle
import zipfile
import io
import struct
import sys
import weakref
import operator
import itertools
import functools

from copy import copy
from functools import lru_cache
from collections import namedtuple
from collections import defaultdict
from enum import Enum
from typing import Optional, Tuple, Union, List, Dict
from typing import Type, NamedTuple, Any

from tqdm import tqdm
from PIL import Image

import numpy as np

DEBUG = int(os.getenv("DEBUG", "0"))

# these are the llops your accelerator must implement, along with toCpu
UnaryOps = Enum("UnaryOps", ["NOOP", "NEG", "RELU", "EXP", "LOG", "SIGN", "RECIPROCAL"])
BinaryOps = Enum("BinaryOps", ["ADD", "SUB", "MUL", "DIV", "POW", "CMPEQ"])
ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
MovementOps = Enum("MovementOps", ["RESHAPE", "PERMUTE", "EXPAND", "FLIP", "STRIDED", "PAD", "SHRINK"])
ProcessingOps = Enum("ProcessingOps", ["CONV"])
LoadOps = Enum("LoadOps", ["FROMCPU", "CONTIGUOUS"])

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps, LoadOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[ProcessingOps], Type[LoadOps]]

GRAPH = int(os.getenv("GRAPH", "0"))

# **** debugging and graphing ****

cnts : Dict[OpType, int] = defaultdict(int)
global_num_max = 0
def log_op(optype : OpType, op : List[Op], ret : DeviceBuffer, inp : List[DeviceBuffer]):
  cnts[optype] += 1
  if DEBUG >= 3:
    print(f"{op} : {', '.join([str(x.shape) for x in inp])} -> {ret.shape}")
  if GRAPH:
    def nm(x):
      global global_num_max
      if not hasattr(x, 'global_num'):
        setattr(x, 'global_num', global_num_max)
        global_num_max += 1
      return f"<<< {x.global_num} >>>"

    top_colors = {LoadOps: '#FFFF80', UnaryOps: "#c0c0c0", ReduceOps: "#8080ff", BinaryOps: "#c0c0c0", MovementOps: "#80ff80", ProcessingOps: "#ff8080"}
    dashed = (optype == LoadOps and hasattr(ret, "_backing")) or (hasattr(ret, "st") and not ret.st.contiguous)  # type: ignore

    for x in inp:
      if len(op) <= 2:
        sop = '.'.join([str(y).split(".")[1] for y in op][::-1])
      elif len(op) <= 4:
        sop = '.'.join([str(y).split(".")[1][0:2] for y in op][::-1])
      else:
        sop = str(len(op))
      G.add_edge(nm(x), nm(ret), label=sop)
      if 'label' not in G.nodes[nm(x)]:
        G.nodes[nm(x)]['label'] = str(x.shape)
    if nm(ret) not in G.nodes:
      G.add_node(nm(ret))

    if optype == ReduceOps:
      G.nodes[nm(ret)]['label'] = str(set(x.shape for x in inp))+"\n"+str(ret.shape)
    else:
      G.nodes[nm(ret)]['label'] = str(ret.shape)
    G.nodes[nm(ret)]['fillcolor'] = (top_colors[optype] + ('80' if dashed else '')) if optype in top_colors else "#ffffff"
    G.nodes[nm(ret)]['style'] = 'filled, dashed' if dashed else 'filled'


class LazyOp(NamedTuple):
  op: Op
  # Any == Union[LazyOp, LazyBuffer, DeviceBuffer]
  src: Tuple[Any, ...]  # type: ignore
  arg: Any = None
  # TODO: add dest to support multiple outputs

# Any == Union[LazyBuffer, DeviceBuffer]
def get_buffers(op:LazyOp) -> List[Any]: return functools.reduce(operator.add, [get_buffers(x) if isinstance(x, LazyOp) else [x] for x in op.src], [])
def get_lazyops(op:LazyOp) -> List[LazyOp]: return functools.reduce(operator.add, [get_lazyops(x) for x in op.src if isinstance(x, LazyOp)], [op])

# a placeholder class to extend by the exec classes
class DeviceBuffer:
  shape: Any   # should be Tuple[int, ...] but ndarray and torch.tensor have incompatible types

# extend this if you don't have an exec_ast function
# used in CPUBuffer and TorchBuffer
class GenericExecAST(DeviceBuffer):
  @classmethod
  def exec_ast(cls, ast:LazyOp, preprocess=lambda x: x):
    srcs = [cls.exec_ast(x, preprocess) if isinstance(x, LazyOp) else preprocess(x) for x in ast.src]
    if ast.op in UnaryOps:
      ret = srcs[0].unary_op(ast.op)
    elif ast.op in BinaryOps:
      assert srcs[0].shape == srcs[1].shape, f"BinaryOps shape mismatch {srcs[0].shape} != {srcs[1].shape}"
      ret = srcs[0].binary_op(ast.op, srcs[1])
    elif ast.op in ReduceOps:
      assert all(r == n or n == 1 for r,n in zip(srcs[0].shape, ast.arg)), f"ReduceOps can't reduce {srcs[0].shape} -> {ast.arg}"
      ret = srcs[0].reduce_op(ast.op, ast.arg)
    elif ast.op in MovementOps:
      ret = srcs[0].movement_op(ast.op, ast.arg)
    elif ast.op in ProcessingOps:
      ret = srcs[0].processing_op(ast.op, srcs[1], ast.arg)
    else:
      raise Exception("unknown op")
    return ret


class CPUBuffer(np.ndarray, GenericExecAST):
  fxn_for_op = {
    UnaryOps.NOOP: lambda x: x[:], UnaryOps.NEG: lambda x: -x, UnaryOps.RELU: lambda x: x.relu(),
    UnaryOps.EXP: lambda x: x.exp(), UnaryOps.LOG: lambda x: x.log(), UnaryOps.SIGN: lambda x: x.sign(), UnaryOps.RECIPROCAL: lambda x: 1.0/x,
    BinaryOps.ADD: operator.add, BinaryOps.SUB: operator.sub, BinaryOps.MUL: operator.mul,
    BinaryOps.DIV: operator.truediv, BinaryOps.POW: operator.pow, BinaryOps.CMPEQ: lambda x,y: (x==y).float(),
    ReduceOps.SUM: lambda x, new_shape: x.sum(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
    ReduceOps.MAX: lambda x, new_shape: x.amax(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
    MovementOps.SHRINK: lambda x, arg: x[tuple(slice(p[0], p[1], None) for p in arg)]
  }

  def relu(x): return np.maximum(x, 0)
  def exp(x): return np.exp(x)
  def log(x): return np.log(x)
  def sign(x): return np.sign(x)
  def float(x): return x.astype(np.float32)
  def flip(x, axis): return np.flip(x, axis)
  def amax(x, *args, **kwargs): return np.amax(x, *args, **kwargs)
  def permute(x, order): return x.transpose(order)
  def pad(x, padding): return np.pad(x, padding).view(CPUBuffer)
  def expand(x, new_shape): return np.broadcast_to(x, new_shape).view(CPUBuffer)
  def strided(x, arg): return np.lib.stride_tricks.as_strided(x.ravel().reshape(x.shape), shape=[y[0] for y in arg], strides=[y[1]*x.dtype.itemsize for y in arg]).view(CPUBuffer)

  @staticmethod
  def fromCPU(x): return x.view(CPUBuffer)
  def toCPU(x): return x

  def unary_op(x, op): return CPUBuffer.fxn_for_op[op](x)
  def binary_op(x, op, y): return CPUBuffer.fxn_for_op[op](x, y)
  def reduce_op(x, op, new_shape): return CPUBuffer.fxn_for_op[op](x, new_shape)
  def movement_op(x, op, arg=None): return CPUBuffer.fxn_for_op[op](x, arg) if op in CPUBuffer.fxn_for_op else getattr(x, op.name.lower())(arg)

  def processing_op(x,op,w,C):
    assert op == ProcessingOps.CONV, f"{op} isn't supported"
    tx = x.movement_op(MovementOps.STRIDED, (
      (C.bs, C.groups*C.cin*x.shape[2]*x.shape[3]), (C.groups, C.cin*x.shape[2]*x.shape[3]),
      (C.oy, C.sy*x.shape[3]), (C.ox, C.sx), (C.cin, x.shape[2]*x.shape[3]), (C.H, C.dy*x.shape[3]), (C.W, C.dx)))
    tw = w.reshape(C.groups, C.rcout, C.cin, C.H, C.W)
    out = np.einsum("nGhwCHW, GkCHW -> nGkhw", tx.ravel().reshape(tx.shape), tw.ravel().reshape(tw.shape))
    return out.reshape(C.bs, C.groups*C.rcout, C.oy, C.ox).view(CPUBuffer)



class GenericShape(GenericExecAST):
  def __init__(self, shape, flops=0): self.shape, self.flops = shape, flops
  def unary_op(self, op:UnaryOps): return GenericShape(self.shape, self.flops + prod(self.shape))
  def binary_op(self, op:BinaryOps, y): return GenericShape(self.shape, self.flops + y.flops + prod(self.shape))
  def reduce_op(self, op:ReduceOps, new_shape:Tuple[int, ...]): return GenericShape(new_shape, self.flops + prod(self.shape))
  def movement_op(self, op:MovementOps, arg): return GenericShape(ShapeTracker(self.shape).movement_op(op, arg).shape, self.flops)
  def processing_op(self, op:ProcessingOps, w, C): return GenericShape(C.out_shape, float("nan"))  # TODO: add flops for this
def get_lazyop_info(ast:LazyOp): return GenericShape.exec_ast(ast, lambda x: GenericShape(x.shape))

# assumes you are using ShapeTracker
# used in GPUBuffer, OpenCLBuffer, and LLVMBuffer
class ExplicitExecAST(DeviceBuffer):
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf=None):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape = self.st.shape

  @classmethod
  def exec_ast(cls, ast:LazyOp): raise NotImplementedError("must be implemented")

  # universal
  def unary_op(self, op:UnaryOps): return type(self)(self.shape).exec_ast(LazyOp(op=op, src=(self,)))
  def binary_op(self, op:BinaryOps, y): return type(self)(self.shape).exec_ast(LazyOp(op=op, src=(self, y)))
  def reduce_op(self, op:ReduceOps, new_shape:Tuple[int, ...]): return type(self)(new_shape).exec_ast(LazyOp(op=op, src=(self,), arg=new_shape))

  # universal for shape tracked
  def movement_op(self, op:MovementOps, arg): return type(self)(ShapeTracker(self.st).movement_op(op, arg), self)
  def contiguous_op(self): return self if self.st.contiguous else self.unary_op(UnaryOps.NOOP)


def divmodidx(acc, d, mod=True):
  lr = f"(idx//{acc})" if acc != 1 else "idx"
  return f"({lr}%{d})" if mod else lr  # don't mod the top shape dimension

@functools.lru_cache(maxsize=None)
def to_shape_strides(shape:Tuple[int, ...], strides:Tuple[int, ...]) -> List[Tuple[int, int]]:
  assert len(shape) == len(strides)
  ret = [(shape[0], strides[0])]
  for i in range(1, len(shape)):
    if (strides[i] != 0 and ret[-1][1] == shape[i]*strides[i]) or ret[-1][0] == 1 or (strides[i] == 0 and ret[-1][1] == 0):
      ret[-1] = (ret[-1][0] * shape[i], strides[i])
    else:
      ret.append((shape[i], strides[i]))
  return ret

class View:
  def __init__(self, shape, strides, offset:int=0):
    self.shape, self.strides, self.offset = tuple(shape), tuple(strides), offset
    self.shape_strides = to_shape_strides(self.shape, self.strides)

  def __repr__(self): return f"View<{self.shape}, {self.strides}, {self.offset}>"

  @functools.cached_property
  def contiguous(self):
    return self.offset == 0 and all(s1 == s2 or s == 1 for s,s1,s2 in zip(self.shape, self.strides, strides_for_shape(self.shape)))

  @functools.cached_property
  def expr(self):
    ret = [f"{self.offset}"] if self.offset != 0 else []
    acc = 1
    for i,(d,s) in enumerate(self.shape_strides[::-1]):
      if d != 1 and s != 0:
        lr = divmodidx(acc, d, i != len(self.shape_strides)-1 and d != prod(self.shape))
        lr = f"({lr}*{s})" if s != 1 else lr
        ret.append(lr)
      acc *= d
    return 'idx=' + ('+'.join(ret) if len(ret) > 0 else "0")

class ZeroView:
  def __init__(self, old_shape, arg):
    self.old_shape, self.arg, self.shape = old_shape, arg, []
    expr, acc = ['valid'], 1
    for s,(x,y) in list(zip(old_shape, arg))[::-1]:
      self.shape = [y-x] + self.shape
      base = divmodidx(acc, self.shape[0], len(self.shape) != len(old_shape)) + f"+{x}"
      expr += ([f"(({base}) >= 0)"] if x < 0 else []) + ([f"(({base}) < {s})"] if y > s else [])
      acc *= self.shape[0]
    self.expr = 'valid=' + ' && '.join(expr)

ViewTypes = Union[View, ZeroView]

@functools.lru_cache(maxsize=None)
def strides_for_shape(shape:Tuple[int, ...]) -> Tuple[int, ...]:
  strides = [1]
  for d in shape[::-1][:-1]:
    strides = [d*strides[0]] + strides
  return tuple(strides)

@functools.lru_cache(maxsize=None)
def view_from_shape(shape:Tuple[int, ...]) -> View:
  assert all(isinstance(x, int) for x in shape) and len(shape) != 0
  return View(tuple(shape), strides_for_shape(shape))

class ShapeTracker:
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]]):
    self.views : List[ViewTypes] = shape.views[:] if isinstance(shape, ShapeTracker) else [view_from_shape(shape)]
  def __repr__(self): return f"{'Complex' if len(self.views) > 1 else ''}ShapeTracker<{self.shape}, {self.views}>"

  @property
  def contiguous(self): return len(self.views) == 1 and self.views[-1].contiguous

  @property
  def shape(self): return self.views[-1].shape

  @property
  def strides(self): return self.views[-1].strides

  @property
  def offset(self): return self.views[-1].offset

  def expr(self): return ';'.join([v.expr for v in self.views[::-1] if v.expr != 'idx=idx' and v.expr != 'valid=valid'])
  def movement_op(self, op, arg):
    getattr(self, str(op).split(".")[1].lower())(*arg)
    return self
  def needs_valid(self): return any(isinstance(v, ZeroView) for v in self.views)

  # TODO: do we really need this for conv?
  # if we replace, confirm the ops taken fold into one view
  def strided(self, *arg):
    view = View([x[0] for x in arg], [x[1] for x in arg])
    # TODO: this does not always require a new view if non contiguous
    if self.contiguous:
      self.views[-1] = view
    else:
      self.views.append(view)

  def reshape(self, *new_shape):
    assert all(isinstance(x, int) for x in new_shape)
    assert prod(self.shape) == prod(new_shape), f"can't reshape {self.shape} -> {new_shape}"

    # check if this is adding or removing 1s (only)
    if tuple([x for x in self.shape if x != 1]) == tuple([x for x in new_shape if x != 1]):
      old_strides = [y for x,y in zip(self.shape, self.strides) if x != 1]
      new_strides = [0 if x == 1 else old_strides.pop(0) for x in new_shape]
      self.views[-1] = View(new_shape, new_strides, self.offset)
      return

    view = View(new_shape, strides_for_shape(new_shape))
    if self.contiguous:
      self.views[-1] = view   # NOTE: if it's contiguous it can't have an offset
    else:
      self.views.append(view)

  def permute(self, *axis):
    assert all(isinstance(x, int) and x >= 0 and x < len(self.shape) for x in axis)
    assert len(set(axis)) == len(axis) and len(axis) == len(self.shape), f"can't permute {self.shape} with {axis}"
    self.views[-1] = View([self.shape[a] for a in axis], [self.strides[a] for a in axis], self.offset)

  # TODO: this is a special case of slice with strides, remove it
  # though it's nice that it can't change size
  def flip(self, *axis): self.stride(*[-1 if i in axis else 1 for i in range(len((self.shape)))])

  # *** under this line are not invertible ***

  # TODO: take this functionality out of slice
  def pad(self, *arg):
    assert all((b>=0 and e>=0) for b,e in arg) and len(arg) == len(self.shape)
    return self.shrink(*[(-b,s+e) for s,(b,e) in zip(self.shape, arg)])

  # TODO: take the pad functionality out of shrink
  def shrink(self, *arg):
    assert len(arg) == len(self.shape)
    offset = sum([self.strides[i]*x for i,(x,_) in enumerate(arg)])
    zeroview = ZeroView(self.shape, arg)
    self.views[-1] = View([y-x for x,y in arg], self.strides, self.offset+offset)
    if zeroview.expr != "valid=valid":
      # if we add a ZeroView, we add another (stock) view also for modding
      self.views += [zeroview, View(self.shape, strides_for_shape(self.shape))]

  def expand(self, *new_shape):
    assert all(isinstance(x, int) for x in new_shape)
    assert all(x == y or x == 1 for x,y in zip(self.shape, new_shape)), f"can't expand {self.shape} into {new_shape}"
    strides = [s if x == y else 0 for s,(x,y) in zip(self.strides, zip(self.shape, new_shape))]
    self.views[-1] = View(new_shape, strides, self.offset)

  # TODO: combine with slice? this doesn't require a ZeroView, though slice shouldn't always either
  def stride(self, *mul):
    assert all(isinstance(x, int) for x in mul)
    strides = [z*m for z,m in zip(self.strides, mul)]
    new_shape = [(s+(abs(m)-1))//abs(m) for s,m in zip(self.shape, mul)]
    offset = sum([(s-1)*z for s,z,m in zip(self.shape, self.strides, mul) if m < 0])
    self.views[-1] = View(new_shape, strides, self.offset + offset)


def dedup(x): return list(dict.fromkeys(x))   # retains list order
def prod(x): return math.prod(x)
def argfix(*x): return tuple() if len(x) == 0 else tuple(x[0]) if isinstance(x[0], tuple) or isinstance(x[0], list) else tuple(x)
def argsort(x): return sorted(range(len(x)), key=x.__getitem__) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python

def reduce_shape(shape, axis): return tuple(1 if i in axis else shape[i] for i in range(len(shape)))
def shape_to_axis(old_shape, new_shape):
  assert len(old_shape) == len(new_shape), "reduce shapes must have same dimensions"
  return tuple([i for i,(a,b) in enumerate(zip(old_shape, new_shape)) if a != b])

ConvArgs = namedtuple('ConvArgs', ['H', 'W', 'groups', 'rcout', 'cin', 'oy', 'ox', 'iy', 'ix', 'sy', 'sx', 'bs', 'cout', 'py', 'py_', 'px', 'px_', 'dy', 'dx', 'out_shape'])
def get_conv_args(x_shape, w_shape, stride=1, groups=1, padding=0, dilation=1, out_shape=None):
  # TODO: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout
  cout,cin,H,W = w_shape
  sy,sx = (stride, stride) if isinstance(stride, int) else stride
  if not isinstance(padding, int) and len(padding) == 4:
    px,px_,py,py_ = padding
  else:
    py,px = (padding, padding) if isinstance(padding, int) else padding
    py_, px_ = py, px
  dy,dx = (dilation, dilation) if isinstance(dilation, int) else dilation
  bs,cin_,iy,ix = x_shape

  # this can change px_ and py_ to make the out_shape right
  # TODO: copy padding names from http://nvdla.org/hw/v1/ias/unit_description.html
  if out_shape is not None:
    py_ = (out_shape[2] - 1) * sy + 1 + dy * (H-1) - iy - py
    px_ = (out_shape[3] - 1) * sx + 1 + dx * (W-1) - ix - px

  # TODO: should be easy to support asymmetric padding by changing output size
  # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html describes these sizes well
  oy = (iy + py + py_ - dy * (H-1) - 1)//sy + 1
  ox = (ix + px + px_ - dx * (W-1) - 1)//sx + 1
  if cin*groups != cin_:
    raise Exception(f"Input Tensor shape {x_shape} does not match the shape of the weights {w_shape}. ({cin*groups} vs. {cin_})")
  assert cout % groups == 0 and (out_shape is None or out_shape == (bs, cout, oy, ox))
  return ConvArgs(H, W, groups, cout//groups, cin, oy, ox, iy, ix, sy, sx, bs, cout, py, py_, px, px_, dy, dx, (bs, cout, oy, ox))

def get_available_llops():
  _buffers, DEFAULT = {}, "CPU"

  _buffers["CPU"] = CPUBuffer
 
  return _buffers, DEFAULT

# lazy can recurse a lot
sys.setrecursionlimit(10000)

OPT = int(os.getenv("OPT", "1"))
NOCONV = int(os.getenv("NOCONV", "0"))

# TODO: movement ops that only change shape are really nops. treat them as such
REMOVE_MOVEMENT_NOPS, MERGE_UNARY_OPS, MERGE_ELEMENTWISE_INTO_REDUCE, SHUFFLE_MOVEMENT_OPS = OPT>=1, OPT>=1, OPT>=1, OPT>=1
MERGE_ELEMENTWISE_OPS, MERGE_ONE_REDUCE_INTO_ELEMENTWISE = OPT>=2, OPT>=2
SHUFFLE_PAD_OPS = OPT>=3  # NOTE: 0/0 is NaN if you pad, so this can change the output

# **** enumerate supported devices ****

class Device:
  _buffers, DEFAULT = get_available_llops()
  for name in _buffers.keys():
    vars()[name] = name

# **** realize helpers ****
def realize_buffers(real_srcs, x):
  if x in real_srcs:
    return realize_buffers(real_srcs, real_srcs[x]) if isinstance(real_srcs[x], LazyOp) else real_srcs[x]
  return LazyOp(x.op, tuple(realize_buffers(real_srcs, y) for y in x.src), x.arg)

# **** realize functions ****
# TODO: make all _realize functions return an AST, perhaps unrealized

def _realize_loadops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], Optional[OpType]]:
  if self.op.op == LoadOps.FROMCPU:
    return Device._buffers[self.device].fromCPU(self.op.arg), [], LoadOps
  elif self.op.op == LoadOps.CONTIGUOUS:
    real_src = self.op.src[0].realize(self.device)
    ret = real_src.contiguous_op()
    return ret, [real_src], LoadOps if ret != real_src else None
  else:
    raise NotImplementedError(f"unknown LoadOp {self.op.op}")

# TODO: these two are generic, replace them?
def _realize_movementops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  real_src = self.op.src[0].realize(self.device)
  return real_src.movement_op(self.op.op, self.op.arg), [real_src], MovementOps

def _realize_processingops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  real_src_x, real_src_w = [x.realize(self.device) for x in self.op.src]
  return real_src_x.processing_op(self.op.op, real_src_w, self.op.arg), [real_src_x, real_src_w], ProcessingOps

# this supports late merging an upstream Elementwise op
def _realize_reduceops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  # TODO: this can also corealize a binary op after the reduce, not just before
  src = self.op.src[0]
  if MERGE_ELEMENTWISE_INTO_REDUCE and src.realized is None and src.optype == BinaryOps and len(src.children) <= 1:
    # this is the new version, deprecate _processing_op
    real_srcs : Dict[LazyBuffer, DeviceBuffer] = {x:x.realize(self.device) for x in get_buffers(src.op)}
    ast = LazyOp(self.op.op, (realize_buffers(real_srcs, src.op),), self.op.arg)
    return self.dbuffer.exec_ast(ast), list(real_srcs.values()), ReduceOps
  else:
    real_src = src.realize(self.device)
    return real_src.reduce_op(self.op.op, self.op.arg), [real_src], ReduceOps

# this supports late merging an upstream Reduce op and even an Elementwise op above that
def _realize_binaryops(self:LazyBuffer) -> Tuple[DeviceBuffer, List[DeviceBuffer], OpType]:
  real_srcs : Dict[LazyBuffer, Union[None, LazyOp, DeviceBuffer]] = {x:None for x in get_buffers(self.op)}
  op_type : OpType = BinaryOps
  psrcs : List[Tuple[LazyBuffer, LazyBuffer]] = [(k,x) for k,x in zip(real_srcs.keys(), map(get_movementroot_contiguous, real_srcs.keys())) if x.optype in [ProcessingOps,ReduceOps] and x.realized is None and len(x.children) <= 1 and len(k.children) <= 1]
  intermediate_shape = self.shape
  if len(psrcs) == 1 and MERGE_ONE_REDUCE_INTO_ELEMENTWISE and (self.device != "OPENCL" or self.shape[-1] == 4):
    if psrcs[0][1].optype == ProcessingOps:
      real_srcs[psrcs[0][0]] = psrcs[0][1].op
      for x in psrcs[0][1].op.src:
        real_srcs[x] = x.realize(self.device)
      op_type = ProcessingOps
    elif psrcs[0][1].optype == ReduceOps:
      src = psrcs[0][1].op.src[0]
      if MERGE_ELEMENTWISE_INTO_REDUCE and src.realized is None and src.optype == BinaryOps and len(src.children) <= 1:
        src = src.op
      real_srcs[psrcs[0][0]] = LazyOp(psrcs[0][1].op.op, (src,), psrcs[0][1].op.arg)
      for x in get_buffers(real_srcs[psrcs[0][0]]):  # type: ignore
        # these are the early buffers
        real_srcs[x] = x.realize(self.device)
      op_type = ReduceOps
    # if the ReduceOp is followed by a reshape, we push this reshape before all the ElementwiseOp inputs
    if psrcs[0][0].shape != psrcs[0][1].shape:
      intermediate_shape = psrcs[0][1].shape
      assert psrcs[0][0].shape == self.shape, f"shape mismatch {psrcs[0][0].shape} != {self.shape}"
  # NOTE: these RESHAPEs will return self if they don't change the shape
  for x in real_srcs.keys():
    if real_srcs[x] is None:
      real_srcs[x] = x.movement_op(MovementOps.RESHAPE, intermediate_shape).realize(self.device)
  ret = self.dbuffer.exec_ast(realize_buffers(real_srcs, self.op))
  return ret.movement_op(MovementOps.RESHAPE, self.shape), [x for x in real_srcs.values() if not isinstance(x, LazyOp) and x is not None], op_type

_realize = {LoadOps:_realize_loadops, ReduceOps:_realize_reduceops, MovementOps:_realize_movementops, BinaryOps:_realize_binaryops, ProcessingOps:_realize_processingops}

# **** lazy operations ****

def get_weakop(op:LazyOp) -> LazyOp: return LazyOp(op.op, tuple(get_weakop(x) if isinstance(x, LazyOp) else weakref.ref(x) for x in op.src), op.arg)
def get_movementroot(root:LazyBuffer) -> LazyBuffer: return get_movementroot(root.op.src[0]) if root.optype == MovementOps and root.realized is None else root
def get_movementroot_contiguous(x:LazyBuffer) -> LazyBuffer: return get_movementroot(x) if x.optype == MovementOps and x.st.contiguous else x

LAZY = int(os.getenv("LAZY", "1"))

class LazyBuffer:
  lazycache : weakref.WeakValueDictionary[LazyOp, LazyBuffer] = weakref.WeakValueDictionary()
  def __new__(cls, device, shape, optype, op):
    # loadops aren't cached
    if optype == LoadOps:
      return super().__new__(cls)
    wop = (device, optype, get_weakop(op))   # NOTE: shape should be deterministic. annoying to cache with the ShapeTracker
    # NOTE: we need "ret" to prevent the new buffer from being immediately deleted
    if wop not in LazyBuffer.lazycache:
      LazyBuffer.lazycache[wop] = ret = super().__new__(cls) # noqa: F841, pylint: disable=W0612
    return LazyBuffer.lazycache[wop]

  def __init__(self, device, shape:Union[ShapeTracker, Tuple[int, ...]], optype:OpType, op:LazyOp):
    if hasattr(self, 'device'):
      return  # cache hit, we return and don't reinit
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape, self.optype, self.op = self.st.shape, optype, op
    self.realized : Optional[DeviceBuffer] = None
    self.device, self.dbuffer = device, Device._buffers[device]
    self.children : weakref.WeakSet[LazyBuffer] = weakref.WeakSet()
    # NOTE: op should be read only after construction of LazyBuffer
    for x in get_buffers(op):
      x.children.add(self)
    if not LAZY:
      self.realize()

  def __repr__(self): return f"<LB {self.shape} op:{self.op.op if self.realized is None else 'realized'}>"

  # this produces a device buffer
  def realize(self:LazyBuffer, required_device=None) -> DeviceBuffer:
    if required_device is not None:
      assert required_device == self.device
    if self.realized is None:
      # we haven't realized the Buffer yet
      self.realized, real_srcs, real_type = _realize[self.optype](self)
      # in lazy mode, we don't log until we realize
      if real_type is not None:
        log_op(real_type, [x.op for x in get_lazyops(self.op)], self.realized, real_srcs)
      # no need to keep the op after realization
      del self.op

    assert self.realized.shape == self.shape
    assert isinstance(self.realized, Device._buffers[self.device])
    return self.realized

  @staticmethod
  def fromCPU(x, device): return LazyBuffer(device, x.shape, LoadOps, LazyOp(LoadOps.FROMCPU, tuple(), x.copy()))
  def toCPU(self): return self.realize().toCPU()

  def unary_op(self:LazyBuffer, op:UnaryOps) -> LazyBuffer: return elementwise_op(op, self)
  def binary_op(self:LazyBuffer, op:BinaryOps, y:LazyBuffer) -> LazyBuffer: return elementwise_op(op, self, y)
  def contiguous_op(self:LazyBuffer) -> LazyBuffer: return LazyBuffer(self.device, self.shape, LoadOps, LazyOp(LoadOps.CONTIGUOUS, (self,)))

  def reduce_op(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[int, ...]) -> LazyBuffer:
    if self.shape == tuple(new_shape):
      return self
    reduce = list(enumerate(zip(self.shape, new_shape)))
    # move the reduce axes to the end
    x = self.movement_op(MovementOps.PERMUTE, [i for i,(s,n) in reduce if s == n] + [i for i,(s,n) in reduce if s != n])
    new_tmp_shape = tuple([n for _,(s,n) in reduce if s == n] + [n for _,(s,n) in reduce if s != n])
    # NOTE: this reshape can only move around 1s
    return LazyBuffer(x.device, new_tmp_shape, ReduceOps, LazyOp(op, (x,), new_tmp_shape)).movement_op(MovementOps.RESHAPE, new_shape)

  # syntactic sugar around PAD and SHRINK
  # TODO: turn RESHAPE into EXPAND and CONTRACT (current EXPAND should be REPEAT)
  def slice(self:LazyBuffer, arg):
    padding = [(max(0, -p[0]), max(0, p[1]-self.shape[i])) for i,p in enumerate(arg)]
    return self.movement_op(MovementOps.PAD, padding).movement_op(MovementOps.SHRINK, tuple((p[0] + padding[i][0], p[1] + padding[i][0]) for i,p in enumerate(arg)))

  def movement_op(self:LazyBuffer, op:MovementOps, arg) -> LazyBuffer:
    # TODO: look into why that copy is needed
    arg = tuple(copy(arg))
    local_st = ShapeTracker(self.shape).movement_op(op, arg)

    # instant nops
    if local_st.contiguous and self.shape == local_st.shape and op != MovementOps.STRIDED:
      return self

    # two ops in a row is one op. merge them if unresolved
    if self.realized is None and self.op.op == op:
      if op in [MovementOps.RESHAPE, MovementOps.EXPAND, MovementOps.SHRINK]:
        return self.op.src[0].movement_op(op, arg)
      if op == MovementOps.PERMUTE:
        return self.op.src[0].movement_op(op, tuple(self.op.arg[i] for i in arg))
      if op == MovementOps.PAD:
        return self.op.src[0].movement_op(op, tuple((b1+b2, e1+e2) for (b1,e1),(b2,e2) in zip(self.op.arg, arg)))
      # TODO: MovementOps.FLIP / MovementOps.STRIDED?

    # some permutes are actually just reshapes
    if op == MovementOps.PERMUTE and local_st.contiguous:
      return self.movement_op(MovementOps.RESHAPE, tuple(self.shape[i] for i in arg))

    # some strideds are actually just reshapes
    # NOTE: due to how strided works, we have to check the parent to be contiguous also
    if op == MovementOps.STRIDED and local_st.contiguous and self.st.contiguous:
      return self.movement_op(MovementOps.RESHAPE, tuple(i for i,_ in arg))

    # if this MovementOp is being applied to a BinaryOp, apply the MovementOp to all the BinaryOp inputs instead
    if SHUFFLE_MOVEMENT_OPS and self.optype == BinaryOps and self.realized is None and len(self.children) == 0 and (SHUFFLE_PAD_OPS or op != MovementOps.PAD) and op not in [MovementOps.EXPAND, MovementOps.STRIDED]:
      def replace_with_movement_op(y:Union[LazyOp, LazyBuffer]) -> LazyBuffer:
        if isinstance(y, LazyBuffer):
          return y.movement_op(op, arg)
        assert y.op in BinaryOps or y.op in UnaryOps
        return elementwise_op(y.op, *[replace_with_movement_op(z) for z in y.src])   # type: ignore
      return replace_with_movement_op(self.op)

    # create the buffer
    ret = LazyBuffer(self.device, ShapeTracker(self.st).movement_op(op, arg), MovementOps, LazyOp(op, (self,), arg))

    # if the ShapeTracker becomes contiguous, replace the whole thing with a reshape (or nothing if shapes match)
    # NOTE: if ret is in the cache, it can already be realized
    if REMOVE_MOVEMENT_NOPS and ret.realized is None and self.realized is None and ret.st.contiguous:
      # MovementOps aren't stacked any more, they each have one parent, find the root
      root = get_movementroot(self)
      if root.st.contiguous and root != self and prod(ret.st.shape) == prod(root.shape):
        return root.movement_op(MovementOps.RESHAPE, ret.st.shape) if ret.st.shape != root.shape else root

    return ret

  def processing_op(self:LazyBuffer, op:ProcessingOps, w:LazyBuffer, C:ConvArgs) -> LazyBuffer:
    x = self
    # TODO: fixup C?
    if NOCONV or not getattr(x.dbuffer, "SUPPORTS_PADDING", False):
      x = x.slice(((0, x.shape[0]), (0, x.shape[1]), (-C.py, x.shape[2]+C.py_), (-C.px, x.shape[3]+C.px_)))

    if NOCONV or not getattr(x.dbuffer, "processing_op", False):
      # universal conv, just mul and reduce
      # TODO: is there any way to replace strided with other movement ops?
      x = x.movement_op(MovementOps.STRIDED, (
        (C.bs, C.groups*C.cin*x.shape[2]*x.shape[3]), (C.groups, C.cin*x.shape[2]*x.shape[3]),
        (1, 1), (C.oy, C.sy*x.shape[3]), (C.ox, C.sx),
        (C.cin, x.shape[2]*x.shape[3]), (C.H, C.dy*x.shape[3]), (C.W, C.dx)))
      #if C.H <= 3 and C.W <= 3:  # max 9x the RAM overhead, this is im2col
      #  x = x.contiguous_op()
      x = x.movement_op(MovementOps.EXPAND, (C.bs, C.groups, C.rcout, C.oy, C.ox, C.cin, C.H, C.W))
      w = w.movement_op(MovementOps.RESHAPE, (1, C.groups, C.rcout, 1, 1, C.cin, C.H, C.W)) \
           .movement_op(MovementOps.EXPAND, (C.bs, C.groups, C.rcout, C.oy, C.ox, C.cin, C.H, C.W))
      return x.binary_op(BinaryOps.MUL, w).reduce_op(ReduceOps.SUM, (C.bs, C.groups, C.rcout, C.oy, C.ox, 1, 1, 1)) \
                                          .movement_op(MovementOps.RESHAPE, (C.bs, C.cout, C.oy, C.ox))
    elif x.device == "OPENCL":
      # TODO: these can be properties on the device buffer
      from accel.opencl.preprocessing import preprocessing_op, postprocessing_op  # type: ignore
      x,w,Cn = preprocessing_op(x, w, C)
      ret = LazyBuffer(x.device, Cn.out_shape, ProcessingOps, LazyOp(op, (x, w), Cn))
      return postprocessing_op(ret, Cn, C)
    else:
      return LazyBuffer(x.device, C.out_shape, ProcessingOps, LazyOp(op, (x, w), C))


def elementwise_op(op:Union[UnaryOps, BinaryOps], *srcs:LazyBuffer) -> LazyBuffer:
  out_device, out_shape = srcs[0].device, srcs[0].shape

  if MERGE_ELEMENTWISE_OPS or (MERGE_UNARY_OPS and len(set(srcs)) == 1):
    # remove the buffers from any (childless) BinaryOps that feed into this
    srcs = tuple(x.op if x.optype == BinaryOps and len(x.children) == 0 and x.realized is None else x for x in srcs)  # type: ignore

  return LazyBuffer(out_device, out_shape, BinaryOps, LazyOp(op, srcs))

class Tensor:
  training, no_grad = False, False

  def __init__(self, data, device=Device.DEFAULT, requires_grad=None):
    if isinstance(data, list):
      data = np.array(data, dtype=np.float32)
    elif isinstance(data, LazyBuffer) and data.device != device:
      # TODO: this has to realize, it shouldn't have to
      data = data.realize().toCPU()

    if isinstance(data, np.ndarray):
      if data.shape == tuple():
        data = data.reshape((1,))
      self.lazydata = LazyBuffer.fromCPU(data.astype(np.float32), device)
    elif isinstance(data, LazyBuffer):
      self.lazydata = data
    else:
      raise Exception(f"can't create Tensor from {data}")

    # tensors have gradients, buffers do not
    self.grad : Optional[Tensor] = None

    # NOTE: this can be in three states. False and None: no gradient, True: gradient
    # None (the default) will be updated to True if it's put in an optimizer
    self.requires_grad : Optional[bool] = requires_grad

    # internal variables used for autograd graph construction
    self._ctx : Optional[Function] = None

  def __repr__(self):
    return f"<Tensor {self.lazydata if self.lazydata.realized is None else self.lazydata.realized!r} with grad {(self.grad.lazydata if self.grad else None)!r}>"

  @property
  def shape(self): return self.lazydata.shape

  # dtype handling was very broken. it's always float32 now
  @property
  def dtype(self): return np.float32

  @property
  def device(self): return self.lazydata.device

  # ***** data handlers ****

  def realize(self):
    self.lazydata.realize()
    return self

  def assign(self, x):
    if not isinstance(x, Tensor):
      x = Tensor(x)
    assert self.shape == x.shape
    self.lazydata = x.lazydata
    return x

  def detach(self): return Tensor(self.lazydata, device=self.device, requires_grad=False)
  def numpy(self): return np.array(self.lazydata.toCPU())

  # TODO: this keeps the legacy behavior working, remove it after refactor
  @property
  def data(self): return self.numpy()

  # TODO: if things are realized this won't work
  def to_(self, device:str):
    assert self.lazydata.realized is None
    self.lazydata.device = device
    if self.grad:
      self.grad.lazydata.device = device

  def to(self, device:str):
    ret = Tensor(self.lazydata, device)
    if self.grad:
      ret.grad = self.grad.to(device)
    return ret

  # ***** creation helper functions *****

  # TODO: remove use of numpy here

  @classmethod
  def zeros(cls, *shape, **kwargs): return cls(np.zeros(shape, dtype=np.float32), **kwargs)

  @classmethod
  def ones(cls, *shape, **kwargs): return cls(np.ones(shape, dtype=np.float32), **kwargs)

  @classmethod
  def empty(cls, *shape, **kwargs): return cls(np.empty(shape, dtype=np.float32), **kwargs)

  @classmethod
  def randn(cls, *shape, **kwargs): return cls(np.random.default_rng().standard_normal(size=shape, dtype=np.float32), **kwargs)

  @classmethod
  def arange(cls, stop, start=0, **kwargs): return cls(np.arange(start=start, stop=stop, dtype=np.float32), **kwargs)

  # TODO: uniform should be a late binding thing
  # Return random number between -1 and 1
  # NOTE: this behavior changed from depending on the shape to not
  @classmethod
  def uniform(cls, *shape, **kwargs): return cls((np.random.default_rng().random(size=shape, dtype=np.float32) * 2 - 1), **kwargs)

  @classmethod
  def scaled_uniform(cls, *shape, **kwargs): return cls((np.random.default_rng().random(size=shape, dtype=np.float32) * 2 - 1) * (prod(shape)**-0.5), **kwargs)

  @classmethod
  # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
  def glorot_uniform(cls, *shape, **kwargs): return cls((np.random.default_rng().random(size=shape, dtype=np.float32) * 2 - 1) * ((6/(shape[0]+prod(shape[1:])))**0.5), **kwargs)

  @classmethod
  def eye(cls, dim, **kwargs): return cls(np.eye(dim, dtype=np.float32), **kwargs)

  # ***** toposort and backward pass *****

  def deepwalk(self):
    def _deepwalk(node, visited, nodes):
      visited.add(node)
      if node._ctx:
        [_deepwalk(i, visited, nodes) for i in node._ctx.parents if i not in visited]
        nodes.append(node)
      return nodes
    return _deepwalk(self, set(), [])

  def backward(self):
    assert self.shape == (1,)

    # fill in the first grad with one
    # this is "implicit gradient creation"
    self.grad = Tensor.ones(*self.shape, device=self.device, requires_grad=False)

    for t0 in reversed(self.deepwalk()):
      if not any(x.requires_grad for x in t0._ctx.parents):
        continue
      assert (t0.grad is not None)
      grads = t0._ctx.backward(t0.grad.lazydata)
      grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
        for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
      for t, g in zip(t0._ctx.parents, grads):
        if g is not None and t.requires_grad:
          assert g.shape == t.shape, f"grad shape must match tensor shape in {self._ctx!r}, {g.shape!r} != {t.shape!r}"
          t.grad = g if t.grad is None else (t.grad + g)
      del t0._ctx

  # ***** non first class ops (hlops) *****

  def __getitem__(self, val):
    arg, new_shape = [], []
    for i, rs in enumerate(val if isinstance(val, (list, tuple)) else [val]) if val is not None else []:
      s = slice(rs, rs+1, None) if isinstance(rs, int) else rs
      arg.append((s.start if s.start is not None else 0, (s.stop if s.stop>=0 else self.shape[i]+s.stop) if s.stop is not None else self.shape[i]))
      assert s.step is None or s.step == 1
      if not isinstance(rs, int):  # don't include in shape if it's an int
        new_shape.append(arg[-1][1] - arg[-1][0])
    new_shape += [self.shape[i] for i in range(len(arg), len(self.shape))]
    return self.slice(arg = arg + [(0,self.shape[i]) for i in range(len(arg), len(self.shape))]).reshape(new_shape if len(new_shape) else (1,))

  def cat(self, *args, dim=0):
    dim = (dim + len(self.shape)) if dim < 0 else dim
    for y in args:
      assert len(y.shape) == len(self.shape) and all(y.shape[i] == s for i,s in enumerate(self.shape) if i != dim)
    args = [self] + list(args)
    shape_cumsum = [0, *itertools.accumulate(y.shape[dim] for y in args)]
    slc = [[(0, s) for s in self.shape] for _ in args]
    for s,k in zip(slc, shape_cumsum):
      s[dim] = (-k, shape_cumsum[-1]-k)
    return functools.reduce(Tensor.__iadd__, [arg.slice(arg=s) for arg,s in zip(args, slc)])

  # TODO: make this nicer with syntactic sugar in slice
  def chunk(self, num, dim):
    slice_params = [[(0, s) for s in self.shape] for _ in range(num)]
    for i,k in enumerate(range(0, self.shape[dim], self.shape[dim]//num)):
      slice_params[i][dim] = (k, min(self.shape[dim], k+self.shape[dim]//num))
    return [self.slice(arg=p) for p in slice_params]

  def matmul(self:Tensor, w:Tensor):
    # NOTE: we use a 1x1 conv2d to do the matmul. mxk @ kxn = (1,k,m,1).conv2d(n,k,1,1)
    bs, groups = prod(self.shape[0:-2]), prod(w.shape[0:-2])
    cin, cout = w.shape[-2], w.shape[-1]
    out_shape_t = tuple(list(self.shape[0:-2])+[cout,-1])
    if len(self.shape) > 1:
      order = tuple(list(range(len(self.shape)-2))+[len(self.shape)-1, len(self.shape)-2])
    else:
      order, out_shape_t = (0,), (cout, )
    worder = tuple(list(range(len(w.shape)-2))+[len(w.shape)-1, len(w.shape)-2])

    # NOTE: with NHWC we can remove the transposes
    # bs x groups*cin x H x W
    cx = self.transpose(order=order).reshape(shape=(bs//groups, groups*cin, -1, 1))
    # groups*cout x cin x H, W
    cw = w.transpose(order=worder).reshape(shape=(groups*cout, cin, 1, 1))
    return cx.conv2d(cw, groups=groups).reshape(shape=out_shape_t).transpose(order=order)

  # TODO: what's the difference between dot and matmul?
  dot = matmul

  # (padding_left, padding_right, padding_top, padding_bottom)
  def pad2d(self, padding:Tuple[int, ...]): return self[:, :, -padding[2]:self.shape[2]+padding[3], -padding[0]:self.shape[3]+padding[1]]
  # TODO: this is totally not transpose
  def transpose(self, order=(1,0)): return self.permute(order=order)
  def flatten(self, start_dim=0): return self.reshape(shape=tuple(list(self.shape[0:start_dim]) + [-1]))

  def _reduce(self, fxn, axis=None, keepdim=False):
    if axis is None:
      axis = range(len(self.shape))
    if isinstance(axis, int):
      axis = [axis]
    axis = tuple([x if x >= 0 else x+len(self.shape) for x in axis])
    shape = [self.shape[i] for i in range(len(self.shape)) if i not in axis]
    ret = fxn(self, axis=axis)
    return ret if keepdim else ret.reshape(shape=[1] if shape == [] else shape)

  def sum(self, axis=None, keepdim=False): return self._reduce(Tensor._sum, axis, keepdim)
  def max(self, axis=None, keepdim=False): return self._reduce(Tensor._max, axis, keepdim)
  def min(self, axis=None, keepdim=False): return -((-self).max(axis=axis, keepdim=keepdim))

  def mean(self, axis=None, keepdim=False):
    out = self.sum(axis=axis, keepdim=keepdim)
    return out * (prod(out.shape)/prod(self.shape))

  def _softmax(self):
    m = self - self.max(axis=len(self.shape)-1, keepdim=True)
    e = m.exp()
    return m, e, e.sum(axis=len(self.shape)-1, keepdim=True)

  def softmax(self):
    _, e, ss = self._softmax()
    return e.div(ss)

  def logsoftmax(self):
    m, _, ss = self._softmax()
    return m - ss.log()

  def dropout(self, p=0.5):
    if not Tensor.training:
      return self
    _mask = np.asarray(np.random.binomial(1, 1.0-p, size=self.shape), dtype=self.dtype)
    return self * Tensor(_mask, requires_grad=False, device=self.device) * (1/(1.0 - p))

  # TODO: support arbitrary strides
  def _pool2d(self, py, px):
    xup = self[:, :, :self.shape[2]-self.shape[2]%py, :self.shape[3]-self.shape[3]%px] if (self.shape[2]%py != 0) or (self.shape[3]%px != 0) else self
    return xup.reshape(shape=(xup.shape[0], xup.shape[1], xup.shape[2]//py, py, xup.shape[3]//px, px))

  def avg_pool2d(self, kernel_size=(2,2)): return self._pool2d(*kernel_size).mean(axis=(3,5))
  def max_pool2d(self, kernel_size=(2,2)): return self._pool2d(*kernel_size).max(axis=(3,5))

  def conv2d(self, weight, bias=None, **kwargs):
    ret = self._conv2d(weight, **kwargs)
    return ret if bias is None else ret.add(bias.reshape(shape=[1, -1, 1, 1]))

  # ***** math functions (unary) *****

  def __neg__(self): return 0.0-self
  def sqrt(self): return self.pow(0.5)
  def square(self): return self*self
  def clip(self, min_, max_): return ((self-min_).relu()+min_) - (self-max_).relu()
  def abs(self): return self.relu() + (-self).relu()
  def sign(self): return self / (self.abs() + 1e-10)

  # ***** activation functions (unary) *****

  def sigmoid(self): return (1.0 + (-self).exp()).reciprocal()
  def elu(self, alpha=1.0): return self.relu() - alpha*(1-self.exp()).relu()
  def swish(self): return self * self.sigmoid()
  silu = swish   # The SiLU function is also known as the swish function.
  def relu6(self): return self.relu() - (self-6).relu()
  def hardswish(self): return self * (self+3).relu6() * (1/6)
  def tanh(self): return 2.0 * ((2.0 * self).sigmoid()) - 1.0
  def gelu(self): return 0.5 * self * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())
  def quick_gelu(self): return self * (self * 1.702).sigmoid()
  def leakyrelu(self, neg_slope=0.01): return self.relu() - (-neg_slope*self).relu()
  def mish(self): return self * self.softplus().tanh()
  def softplus(self, limit=20, beta=1): return (1/beta) * (1 + (self*beta).exp()).log()

  # ***** broadcasted binary ops *****

  @staticmethod
  def broadcasted(fxn, x, y):
    tt = [arg for arg in [x,y] if isinstance(arg, Tensor)][0]  # this is the prototype tensor
    x,y = [Tensor([t], device=tt.device, requires_grad=False) if not isinstance(t, Tensor) else t for t in [x,y]]
    x,y = [t.reshape([1]*(max(len(x.shape), len(y.shape))-len(t.shape)) + list(t.shape)) for t in [x,y]]
    shape_ret = tuple(max(sx, sy) for sx,sy in zip(x.shape, y.shape))
    return fxn(x.expand(shape_ret), y.expand(shape_ret))

  # TODO: are these the only ones that can take number arguments?
  def add(self, x): return Tensor.broadcasted(Tensor._add, self, x)
  def sub(self, x): return Tensor.broadcasted(Tensor._sub, self, x)
  def mul(self, x): return Tensor.broadcasted(Tensor._mul, self, x)
  def pow(self, x): return Tensor.broadcasted(Tensor._pow, self, x)
  def div(self, y): return self * (y.reciprocal() if isinstance(y, Tensor) else (1/y))

  # ***** functional nn ops *****

  # TODO: fix the kwargs problem, then remove these (or not, since they now fix tuples)
  def reshape(self, shape, *args): return self._reshape(shape=argfix(shape, *args))
  def expand(self, shape, *args): return self._expand(shape=argfix(shape, *args))
  def permute(self, order, *args): return self._permute(order=argfix(order, *args))

  def linear(self, weight:Tensor, bias:Optional[Tensor]=None):
    x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
    return x.add(bias) if bias is not None else x

  def sequential(self, ll:List[Callable[[Tensor], Tensor]]): return functools.reduce(lambda x,f: f(x), ll, self)

  def layernorm(self, axis=-1, eps=1e-5):
    y = (self - self.mean(axis=axis, keepdim=True))
    return y.div((y*y).mean(axis=axis, keepdim=True).add(eps).sqrt())


# An instantiation of the Function is the Context
class Function:
  def __init__(self, device:str, *tensors:Tensor):
    self.device, self.parents = device, tensors
    self.needs_input_grad = [t.requires_grad for t in self.parents]
    self.requires_grad = True if any(self.needs_input_grad) else (None if any(x is None for x in self.needs_input_grad) else False)
    self.saved_tensors : List[Tensor] = []

  def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
  def backward(self, *args, **kwargs): raise NotImplementedError(f"backward not implemented for {type(self)}")

  # NOTE: it doesn't hurt to save this since the ctx will be freed fast without grad
  def save_for_backward(self, *x): self.saved_tensors.extend(x)

  @classmethod
  def apply(cls, *x:Tensor, **kwargs):
    ctx = cls(x[0].device, *x)
    ret = Tensor(ctx.forward(*[t.lazydata for t in x], **kwargs), device=ctx.device, requires_grad=ctx.requires_grad)
    if ctx.requires_grad and not Tensor.no_grad:
      ret._ctx = ctx    # used by autograd engine
    return ret

class ReLU(Function):
  def forward(self, x):
    ret = x.unary_op(UnaryOps.RELU)
    self.save_for_backward(ret)
    return ret

  def backward(self, grad_output):
    return self.saved_tensors[0].unary_op(UnaryOps.SIGN).binary_op(BinaryOps.MUL, grad_output)

class Log(Function):
  def forward(self, x):
    self.save_for_backward(x)
    return x.unary_op(UnaryOps.LOG)

  def backward(self, grad_output):
    return grad_output.binary_op(BinaryOps.DIV, self.saved_tensors[0])

class Exp(Function):
  def forward(self, x):
    ret = x.unary_op(UnaryOps.EXP)
    self.save_for_backward(ret)
    return ret

  def backward(self, grad_output):
    return self.saved_tensors[0].binary_op(BinaryOps.MUL, grad_output)

class Reciprocal(Function):
  def forward(self, x):
    ret = x.unary_op(UnaryOps.RECIPROCAL)
    self.save_for_backward(ret)
    return ret

  def backward(self, grad_output):
    return grad_output.unary_op(UnaryOps.NEG).binary_op(BinaryOps.MUL, self.saved_tensors[0]).binary_op(BinaryOps.MUL, self.saved_tensors[0])

# TODO: add Neg? confirm the optimizer on Sub good enough

# ************* reduce ops *************

class Sum(Function):
  def forward(self, x, axis=None):
    self.input_shape = x.shape
    return x.reduce_op(ReduceOps.SUM, reduce_shape(x.shape, axis))

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.EXPAND, self.input_shape)

class Max(Function):
  def forward(self, x, axis=None):
    ret = x.reduce_op(ReduceOps.MAX, reduce_shape(x.shape, axis))
    self.save_for_backward(x, ret)
    return ret

  def backward(self, grad_output):
    x, ret = self.saved_tensors

    # 1s in locations where the max was chosen (can be two locations)
    max_is_1s = x.binary_op(BinaryOps.CMPEQ, ret.movement_op(MovementOps.EXPAND, x.shape))

    # sum of locations, averaged
    div = max_is_1s.reduce_op(ReduceOps.SUM, grad_output.shape)
    div = div.movement_op(MovementOps.EXPAND, x.shape)
    max_is_amount = max_is_1s.binary_op(BinaryOps.DIV, div)

    grad_output_expanded = grad_output.movement_op(MovementOps.EXPAND, x.shape)
    return max_is_amount.binary_op(BinaryOps.MUL, grad_output_expanded)

# ************* binary ops *************

class Add(Function):
  def forward(self, x, y):
    return x.binary_op(BinaryOps.ADD, y)

  def backward(self, grad_output):
    return grad_output if self.needs_input_grad[0] else None, \
           grad_output if self.needs_input_grad[1] else None

class Sub(Function):
  def forward(self, x, y):
    return x.binary_op(BinaryOps.SUB, y)

  def backward(self, grad_output):
    return grad_output if self.needs_input_grad[0] else None, \
           grad_output.unary_op(UnaryOps.NEG) if self.needs_input_grad[1] else None

class Mul(Function):
  def forward(self, x, y):
    self.save_for_backward(x, y)
    return x.binary_op(BinaryOps.MUL, y)

  def backward(self, grad_output):
    return self.saved_tensors[1].binary_op(BinaryOps.MUL, grad_output) if self.needs_input_grad[0] else None, \
           self.saved_tensors[0].binary_op(BinaryOps.MUL, grad_output) if self.needs_input_grad[1] else None

class Pow(Function):
  def forward(self, x, y):
    ret = x.binary_op(BinaryOps.POW, y)
    self.save_for_backward(x, y, ret)
    return ret

  def backward(self, grad_output):
    x,y,powxy = self.saved_tensors
    # grad_x = grad_output * y * (pow(x,y)/x)
    # grad_y = grad_output * log(x) * pow(x,y)
    return grad_output.binary_op(BinaryOps.MUL, y.binary_op(BinaryOps.MUL, powxy.binary_op(BinaryOps.DIV, x))) if self.needs_input_grad[0] else None, \
           grad_output.binary_op(BinaryOps.MUL, x.unary_op(UnaryOps.LOG).binary_op(BinaryOps.MUL, powxy)) if self.needs_input_grad[1] else None

# ************* movement ops *************

# NOTE: this is sum in reverse
class Expand(Function):
  def forward(self, x, shape):
    self.input_shape = x.shape
    return x.movement_op(MovementOps.EXPAND, shape)

  def backward(self, grad_output):
    return grad_output.reduce_op(ReduceOps.SUM, self.input_shape)

class Reshape(Function):
  def forward(self, x, shape):
    self.input_shape = x.shape
    shape = tuple(-prod(x.shape) // prod(shape) if s == -1 else s for s in shape)
    return x.movement_op(MovementOps.RESHAPE, shape)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.RESHAPE, self.input_shape)

class Permute(Function):
  def forward(self, x, order=(1,0)):
    self.input_order = order
    return x.movement_op(MovementOps.PERMUTE, order)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.PERMUTE, tuple(argsort(self.input_order)))

# TODO: merge Slice and Flip into Stride with the 3 arguments
class Slice(Function):
  def forward(self, x, arg=None):
    self.narg = tuple((0-p[0], x.shape[i]-p[0]) for i,p in enumerate(arg))
    return x.slice(tuple(arg))

  def backward(self, grad_output):
    return grad_output.slice(self.narg)

class Flip(Function):
  def forward(self, x, axis):
    self.axis = axis
    return x.movement_op(MovementOps.FLIP, axis)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.FLIP, self.axis)

# ************* processing ops *************

class Conv2D(Function):
  def forward(self, x, w, stride=1, groups=1, dilation=1, padding=0):
    self.C = get_conv_args(x.shape, w.shape, stride, groups, dilation=dilation, padding=padding)
    self.save_for_backward(x,w)
    return x.processing_op(ProcessingOps.CONV, w, self.C)

  def backward(self, grad_output):
    x, w = self.saved_tensors
    C = self.C   # conv args from the context
    dx, dw = None, None

    if self.needs_input_grad[0]:    # compute derivative of inputs using ProcessingOps.CONV (this is a transposed conv)
      xt = grad_output
      if C.sx > 1 or C.sy > 1:   # unstride. NOTE: this is really memory intensive for big strides. (but only when we contiguous it)
        xt = xt.movement_op(MovementOps.RESHAPE, (grad_output.shape[0], grad_output.shape[1], grad_output.shape[2], 1, grad_output.shape[3], 1))
        xt = xt.movement_op(MovementOps.PAD, ((0,0), (0,0), (0,0), (0,C.sy-1), (0,0), (0,C.sx-1)))
        xt = xt.movement_op(MovementOps.RESHAPE, (xt.shape[0], xt.shape[1], xt.shape[2]*C.sy, xt.shape[4]*C.sx))
      wt = w.movement_op(MovementOps.RESHAPE, (C.groups, C.rcout, C.cin, C.H, C.W)).movement_op(MovementOps.PERMUTE, (0, 2, 1, 3, 4)).movement_op(MovementOps.FLIP, (3, 4))
      wt = wt.movement_op(MovementOps.RESHAPE, (C.groups*C.cin, C.rcout, C.H, C.W))
      py, px = (C.H-1)*C.dy - C.py, (C.W-1)*C.dx - C.px
      Cdx = get_conv_args(xt.shape, wt.shape, out_shape=x.shape, dilation=(C.dy, C.dx), padding=(py, px), groups=C.groups)
      dx = xt.processing_op(ProcessingOps.CONV, wt, Cdx)

    if self.needs_input_grad[1]:   # compute derivative of weights using ProcessingOps.CONV
      xdw = x.movement_op(MovementOps.RESHAPE, (C.bs, C.groups, C.cin, C.iy, C.ix)).movement_op(MovementOps.PERMUTE, (2, 1, 0, 3, 4))
      xdw = xdw.movement_op(MovementOps.RESHAPE, (C.cin, C.groups*C.bs, C.iy, C.ix))
      grad_output_dw = grad_output.movement_op(MovementOps.PERMUTE, (1,0,2,3))
      Cdw = get_conv_args(xdw.shape, grad_output_dw.shape, out_shape=(w.shape[1], w.shape[0], w.shape[2], w.shape[3]), padding=(C.py, C.px), stride=(C.dy, C.dx), dilation=(C.sy, C.sx), groups=C.groups)
      dw = xdw.processing_op(ProcessingOps.CONV, grad_output_dw, Cdw).movement_op(MovementOps.PERMUTE, (1,0,2,3))

    return dx, dw

# register functions to move between devices
for device in [device for device in Device.__dict__.keys() if device[0] != "_"]:
  setattr(Tensor, f"{device.lower()}", functools.partialmethod(Tensor.to, Device.__dict__[device]))
  setattr(Tensor, f"{device.lower()}_", functools.partialmethod(Tensor.to_, Device.__dict__[device]))

# register all the mlops "math" operations
def register(name:str, fxn:Function):
  setattr(Tensor, "_"+name if hasattr(Tensor, name) else name, functools.partialmethod(fxn.apply))
for name, cls in (
         ("ReLU", ReLU),
         ("Log", Log),
         ("Exp", Exp),
         ("Reciprocal", Reciprocal),
         ("Sum", Sum),
         ("Max", Max),
         ("Add", Add),
         ("Sub", Sub),
         ("Mul", Mul),
         ("Pow", Pow),
         ("Expand", Expand),
         ("Reshape", Reshape),
         ("Permute", Permute),
         ("Slice", Slice),
         ("Flip", Flip),
         ("Conv2D", Conv2D),
        ):
  if name[0] != "_" and name != "Function" and not name.endswith("Ops"):
    register(name.lower(), cls)

# register the operators
def register_op(name, fxn):
  setattr(Tensor, f"__{name}__", fxn)
  setattr(Tensor, f"__i{name}__", lambda self,x: self.assign(fxn(self,x)))
  setattr(Tensor, f"__r{name}__", lambda self,x: fxn(x,self))
for name in ['add', 'sub', 'mul', 'pow', 'matmul', 'truediv']:
  register_op(name, getattr(Tensor, name if name != 'truediv' else 'div'))


def get_parameters(obj):
  parameters = []
  if isinstance(obj, Tensor):
    parameters.append(obj)
  elif isinstance(obj, list) or isinstance(obj, tuple):
    for x in obj:
      parameters.extend(get_parameters(x))
  elif hasattr(obj, '__dict__'):
    for v in obj.__dict__.values():
      parameters.extend(get_parameters(v))
  return parameters

class Optimizer:
  def __init__(self, params):
    # if it's None, but being put into an optimizer, set it to True
    for x in params:
      if x.requires_grad is None:
        x.requires_grad = True

    self.params = [x for x in params if x.requires_grad]

  # TODO: this probably shouldn't change the gradients, just the ones used by the optimizer
  def clipnorm(self, amount=1):
    for param in self.params:
      # clipnorm is the L2 norm, not value: is this right?
      param.grad.assign(param.grad.clip(-(amount**2), (amount**2)))

  def zero_grad(self):
    for param in self.params:
      param.grad = None

  def realize(self, extra=None):
    # TODO: corealize
    for p in self.params + extra if extra is not None else self.params:
      p.realize()

def prod(x): return math.prod(x)

class Conv2d:
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (kernel_size[0], kernel_size[1])
    self.stride = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])
    self.padding = (padding, ) * 4 if isinstance(padding, int) else ((padding[0], padding[0], padding[1], padding[1]) if len(padding) == 2 else padding)
    self.weight = Tensor.glorot_uniform(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
    self.bias = Tensor.zeros(out_channels) if bias else None

  def __call__(self, x):
    return x.conv2d(self.weight, self.bias, padding=self.padding, stride=self.stride)

class Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = Tensor.glorot_uniform(out_features, in_features)
    self.bias = Tensor.zeros(out_features) if bias else None

  def __call__(self, x):
    return x.linear(self.weight.transpose(), self.bias)


def my_unpickle(fb0):
  key_prelookup = {}
  class HackTensor:
    def __new__(cls, *args):
      #print(args)
      ident, storage_type, obj_key, location, obj_size = args[0][0:5]
      assert ident == 'storage'

      assert prod(args[2]) == obj_size
      ret = np.zeros(args[2], dtype=storage_type)
      key_prelookup[obj_key] = (storage_type, obj_size, ret, args[2], args[3])
      return ret

  class HackParameter:
    def __new__(cls, *args):
      #print(args)
      pass

  class Dummy:
    pass

  class MyPickle(pickle.Unpickler):
    def find_class(self, module, name):
      #print(module, name)
      if name == 'FloatStorage':
        return np.float32
      if name == 'LongStorage':
        return np.int64
      if name == 'HalfStorage':
        return np.float16
      if module == "torch._utils":
        if name == "_rebuild_tensor_v2":
          return HackTensor
        elif name == "_rebuild_parameter":
          return HackParameter
      else:
        try:
          return pickle.Unpickler.find_class(self, module, name)
        except Exception:
          return Dummy

    def persistent_load(self, pid):
      return pid

  return MyPickle(fb0).load(), key_prelookup

def fake_torch_load_zipped(fb0, load_weights=True):
  with zipfile.ZipFile(fb0, 'r') as myzip:
    with myzip.open('archive/data.pkl') as myfile:
      ret = my_unpickle(myfile)
    if load_weights:
      for k,v in ret[1].items():
        with myzip.open(f'archive/data/{k}') as myfile:
          if v[2].dtype == "object":
            print(f"issue assigning object on {k}")
            continue
          np.copyto(v[2], np.frombuffer(myfile.read(), v[2].dtype).reshape(v[3]))
  return ret[0]

def fake_torch_load(b0):
  # convert it to a file
  fb0 = io.BytesIO(b0)

  if b0[0:2] == b"\x50\x4b":
    return fake_torch_load_zipped(fb0)

  # skip three junk pickles
  pickle.load(fb0)
  pickle.load(fb0)
  pickle.load(fb0)

  ret, key_prelookup = my_unpickle(fb0)

  # create key_lookup
  key_lookup = pickle.load(fb0)
  key_real = [None] * len(key_lookup)
  for k,v in key_prelookup.items():
    key_real[key_lookup.index(k)] = v

  # read in the actual data
  for storage_type, obj_size, np_array, np_shape, np_strides in key_real:
    ll = struct.unpack("Q", fb0.read(8))[0]
    assert ll == obj_size
    bytes_size = {np.float32: 4, np.int64: 8}[storage_type]
    mydat = fb0.read(ll * bytes_size)
    np.copyto(np_array, np.frombuffer(mydat, storage_type).reshape(np_shape))

    # numpy stores its strides in bytes
    real_strides = tuple([x*bytes_size for x in np_strides])
    np_array.strides = real_strides

  return ret

def get_child(parent, key):
  obj = parent
  for k in key.split('.'):
    if k.isnumeric():
      obj = obj[int(k)]
    elif isinstance(obj, dict):
      obj = obj[k]
    else:
      obj = getattr(obj, k)
  return obj

# TODO: refactor AttnBlock, CrossAttention, CLIPAttention to share code

# TODO: rename to GroupNorm and put in nn.py
class Normalize:
  def __init__(self, in_channels, num_groups=32):
    self.weight = Tensor.empty(in_channels)
    self.bias = Tensor.empty(in_channels)
    self.num_groups = num_groups

  def __call__(self, x):
    # reshape for layernorm to work as group norm
    # subtract mean and divide stddev
    if self.num_groups == None:  # just layernorm
      x = x.layernorm()
    else:
      x = x.reshape(x.shape[0], self.num_groups, -1).layernorm().reshape(x.shape)

    # elementwise_affine on channels
    if len(x.shape) == 4:
      # HACK for channels in conv
      return (x * self.weight.reshape(1, -1, 1, 1)) + self.bias.reshape(1, -1, 1, 1)
    else:
      return x.linear(self.weight, self.bias)

class AttnBlock:
  def __init__(self, in_channels):
    self.norm = Normalize(in_channels)
    self.q = Conv2d(in_channels, in_channels, 1)
    self.k = Conv2d(in_channels, in_channels, 1)
    self.v = Conv2d(in_channels, in_channels, 1)
    self.proj_out = Conv2d(in_channels, in_channels, 1)

  # copied from AttnBlock in ldm repo
  def __call__(self, x):
    h_ = self.norm(x)
    q,k,v = self.q(h_), self.k(h_), self.v(h_)

    # compute attention
    b,c,h,w = q.shape
    q = q.reshape(b,c,h*w)
    q = q.permute(0,2,1)   # b,hw,c
    k = k.reshape(b,c,h*w) # b,c,hw
    w_ = q @ k
    w_ = w_ * (c**(-0.5))
    w_ = w_.softmax()

    # attend to values
    v = v.reshape(b,c,h*w)
    w_ = w_.permute(0,2,1)
    h_ = v @ w_
    h_ = h_.reshape(b,c,h,w)

    return x + self.proj_out(h_)

class ResnetBlock:
  def __init__(self, in_channels, out_channels=None):
    self.norm1 = Normalize(in_channels)
    self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
    self.norm2 = Normalize(out_channels)
    self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
    self.nin_shortcut = Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else lambda x: x

  def __call__(self, x):
    h = self.conv1(self.norm1(x).swish())
    h = self.conv2(self.norm2(h).swish())
    return self.nin_shortcut(x) + h

class Mid:
  def __init__(self, block_in):
    self.block_1 = ResnetBlock(block_in, block_in)
    self.attn_1 = AttnBlock(block_in)
    self.block_2 = ResnetBlock(block_in, block_in)

  def __call__(self, x):
    return x.sequential([self.block_1, self.attn_1, self.block_2])

class Decoder:
  def __init__(self):
    sz = [(128, 256), (256, 512), (512, 512), (512, 512)]
    self.conv_in = Conv2d(4,512,3, padding=1)
    self.mid = Mid(512)

    arr = []
    for i,s in enumerate(sz):
      arr.append({"block":
        [ResnetBlock(s[1], s[0]),
         ResnetBlock(s[0], s[0]),
         ResnetBlock(s[0], s[0])]})
      if i != 0: arr[-1]['upsample'] = {"conv": Conv2d(s[0], s[0], 3, padding=1)}
    self.up = arr

    self.norm_out = Normalize(128)
    self.conv_out = Conv2d(128, 3, 3, padding=1)

  def __call__(self, x):
    x = self.conv_in(x)
    x = self.mid(x)

    for l in self.up[::-1]:
      print("decode", x.shape)
      for b in l['block']: x = b(x)
      if 'upsample' in l:
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html ?
        bs,c,py,px = x.shape
        x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
        x = l['upsample']['conv'](x)
      x.realize()

    return self.conv_out(self.norm_out(x).swish())


class Encoder:
  def __init__(self):
    sz = [(128, 128), (128, 256), (256, 512), (512, 512)]
    self.conv_in = Conv2d(3,128,3, padding=1)

    arr = []
    for i,s in enumerate(sz):
      arr.append({"block":
        [ResnetBlock(s[0], s[1]),
         ResnetBlock(s[1], s[1])]})
      if i != 3: arr[-1]['downsample'] = {"conv": Conv2d(s[1], s[1], 3, stride=2, padding=(0,1,0,1))}
    self.down = arr

    self.mid = Mid(512)
    self.norm_out = Normalize(512)
    self.conv_out = Conv2d(512, 8, 3, padding=1)

  def __call__(self, x):
    x = self.conv_in(x)

    for l in self.down:
      print("encode", x.shape)
      for b in l['block']: x = b(x)
      if 'downsample' in l: x = l['downsample']['conv'](x)

    x = self.mid(x)
    return self.conv_out(self.norm_out(x).swish())

class AutoencoderKL:
  def __init__(self):
    self.encoder = Encoder()
    self.decoder = Decoder()
    self.quant_conv = Conv2d(8, 8, 1)
    self.post_quant_conv = Conv2d(4, 4, 1)

  def __call__(self, x):
    latent = self.encoder(x)
    latent = self.quant_conv(latent)
    latent = latent[:, 0:4]  # only the means
    print("latent", latent.shape)
    latent = self.post_quant_conv(latent)
    return self.decoder(latent)

# not to be confused with ResnetBlock
class ResBlock:
  def __init__(self, channels, emb_channels, out_channels):
    self.in_layers = [
      Normalize(channels),
      Tensor.silu,
      Conv2d(channels, out_channels, 3, padding=1)
    ]
    self.emb_layers = [
      Tensor.silu,
      Linear(emb_channels, out_channels)
    ]
    self.out_layers = [
      Normalize(out_channels),
      Tensor.silu,
      lambda x: x,
      Conv2d(out_channels, out_channels, 3, padding=1)
    ]
    self.skip_connection = Conv2d(channels, out_channels, 1) if channels != out_channels else lambda x: x

  def __call__(self, x, emb):
    h = x.sequential(self.in_layers)
    emb_out = emb.sequential(self.emb_layers)
    h = h + emb_out.reshape(*emb_out.shape, 1, 1)
    h = h.sequential(self.out_layers)
    ret = self.skip_connection(x) + h
    return ret

class CrossAttention:
  def __init__(self, query_dim, context_dim, n_heads, d_head):
    self.to_q = Linear(query_dim, n_heads*d_head, bias=False)
    self.to_k = Linear(context_dim, n_heads*d_head, bias=False)
    self.to_v = Linear(context_dim, n_heads*d_head, bias=False)
    self.scale = d_head ** -0.5
    self.num_heads = n_heads
    self.head_size = d_head
    self.to_out = [Linear(n_heads*d_head, query_dim)]

  def __call__(self, x, context=None):
    context = x if context is None else context
    q,k,v = self.to_q(x), self.to_k(context), self.to_v(context)
    q = q.reshape(x.shape[0], -1, self.num_heads, self.head_size).permute(0,2,1,3)  # (bs, num_heads, time, head_size)
    k = k.reshape(x.shape[0], -1, self.num_heads, self.head_size).permute(0,2,3,1)  # (bs, num_heads, head_size, time)
    v = v.reshape(x.shape[0], -1, self.num_heads, self.head_size).permute(0,2,1,3)  # (bs, num_heads, time, head_size)

    score = q.dot(k) * self.scale
    weights = score.softmax()                     # (bs, num_heads, time, time)
    attention = weights.dot(v).permute(0,2,1,3)   # (bs, time, num_heads, head_size)

    h_ = attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size))
    return h_.sequential(self.to_out)

class GEGLU:
  def __init__(self, dim_in, dim_out):
    self.proj = Linear(dim_in, dim_out * 2)
    self.dim_out = dim_out

  def __call__(self, x):
    x, gate = self.proj(x).chunk(2, dim=-1)
    return x * gate.gelu()

class FeedForward:
  def __init__(self, dim, mult=4):
    self.net = [
      GEGLU(dim, dim*mult),
      lambda x: x,
      Linear(dim*mult, dim)
    ]

  def __call__(self, x):
    return x.sequential(self.net)

class BasicTransformerBlock:
  def __init__(self, dim, context_dim, n_heads, d_head):
    self.attn1 = CrossAttention(dim, dim, n_heads, d_head)
    self.ff = FeedForward(dim)
    self.attn2 = CrossAttention(dim, context_dim, n_heads, d_head)
    self.norm1 = Normalize(dim, num_groups=None)
    self.norm2 = Normalize(dim, num_groups=None)
    self.norm3 = Normalize(dim, num_groups=None)

  def __call__(self, x, context=None):
    x = self.attn1(self.norm1(x)) + x
    x = self.attn2(self.norm2(x), context=context) + x
    x = self.ff(self.norm3(x)) + x
    return x

class SpatialTransformer:
  def __init__(self, channels, context_dim, n_heads, d_head):
    self.norm = Normalize(channels)
    assert channels == n_heads * d_head
    self.proj_in = Conv2d(channels, n_heads * d_head, 1)
    self.transformer_blocks = [BasicTransformerBlock(channels, context_dim, n_heads, d_head)]
    self.proj_out = Conv2d(n_heads * d_head, channels, 1)

  def __call__(self, x, context=None):
    b, c, h, w = x.shape
    x_in = x
    x = self.norm(x)
    x = self.proj_in(x)
    x = x.reshape(b, c, h*w).permute(0,2,1)
    for block in self.transformer_blocks:
      x = block(x, context=context)
    x = x.permute(0,2,1).reshape(b, c, h, w)
    ret = self.proj_out(x) + x_in
    return ret

class Downsample:
  def __init__(self, channels):
    self.op = Conv2d(channels, channels, 3, stride=2, padding=1)

  def __call__(self, x):
    return self.op(x)

class Upsample:
  def __init__(self, channels):
    self.conv = Conv2d(channels, channels, 3, padding=1)

  def __call__(self, x):
    bs,c,py,px = x.shape
    x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
    return self.conv(x)

def timestep_embedding(timesteps, dim, max_period=10000):
  half = dim // 2
  freqs = np.exp(-math.log(max_period) * np.arange(0, half, dtype=np.float32) / half)
  args = timesteps.numpy() * freqs
  embedding = np.concatenate([np.cos(args), np.sin(args)])
  return Tensor(embedding).reshape(1, -1)

class UNetModel:
  def __init__(self):
    self.time_embed = [
      Linear(320, 1280),
      Tensor.silu,
      Linear(1280, 1280),
    ]
    self.input_blocks = [
      [Conv2d(4, 320, kernel_size=3, padding=1)],
      # TODO: my head sizes and counts are a guess
      [ResBlock(320, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
      [ResBlock(320, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
      [Downsample(320)],
      [ResBlock(320, 1280, 640), SpatialTransformer(640, 768, 8, 80)],
      [ResBlock(640, 1280, 640), SpatialTransformer(640, 768, 8, 80)],
      [Downsample(640)],
      [ResBlock(640, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
      [ResBlock(1280, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
      [Downsample(1280)],
      [ResBlock(1280, 1280, 1280)],
      [ResBlock(1280, 1280, 1280)]
    ]
    self.middle_block = [
      ResBlock(1280, 1280, 1280),
      SpatialTransformer(1280, 768, 8, 160),
      ResBlock(1280, 1280, 1280)
    ]
    self.output_blocks = [
      [ResBlock(2560, 1280, 1280)],
      [ResBlock(2560, 1280, 1280)],
      [ResBlock(2560, 1280, 1280), Upsample(1280)],
      [ResBlock(2560, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
      [ResBlock(2560, 1280, 1280), SpatialTransformer(1280, 768, 8, 160)],
      [ResBlock(1920, 1280, 1280), SpatialTransformer(1280, 768, 8, 160), Upsample(1280)],
      [ResBlock(1920, 1280, 640), SpatialTransformer(640, 768, 8, 80)],  # 6
      [ResBlock(1280, 1280, 640), SpatialTransformer(640, 768, 8, 80)],
      [ResBlock(960, 1280, 640), SpatialTransformer(640, 768, 8, 80), Upsample(640)],
      [ResBlock(960, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
      [ResBlock(640, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
      [ResBlock(640, 1280, 320), SpatialTransformer(320, 768, 8, 40)],
    ]
    self.out = [
      Normalize(320),
      Tensor.silu,
      Conv2d(320, 4, kernel_size=3, padding=1)
    ]

  def __call__(self, x, timesteps=None, context=None):
    # TODO: real time embedding
    t_emb = timestep_embedding(timesteps, 320)
    emb = t_emb.sequential(self.time_embed)

    def run(x, bb):
      if isinstance(bb, ResBlock): x = bb(x, emb)
      elif isinstance(bb, SpatialTransformer): x = bb(x, context)
      else: x = bb(x)
      return x

    saved_inputs = []
    for i,b in enumerate(self.input_blocks):
      #print("input block", i)
      for bb in b:
        x = run(x, bb)
      saved_inputs.append(x)
      x.realize()
    for bb in self.middle_block:
      x = run(x, bb)
    for i,b in enumerate(self.output_blocks):
      #print("output block", i)
      x = x.cat(saved_inputs.pop(), dim=1)
      for bb in b:
        x = run(x, bb)
      x.realize()
    return x.sequential(self.out)

class CLIPMLP:
  def __init__(self):
    self.fc1 = Linear(768, 3072)
    self.fc2 = Linear(3072, 768)

  def __call__(self, hidden_states):
    hidden_states = self.fc1(hidden_states)
    hidden_states = hidden_states.quick_gelu()
    hidden_states = self.fc2(hidden_states)
    return hidden_states

class CLIPAttention:
  def __init__(self):
    self.embed_dim = 768
    self.num_heads = 12
    self.head_dim = self.embed_dim // self.num_heads
    self.scale = self.head_dim**-0.5
    self.k_proj = Linear(self.embed_dim, self.embed_dim)
    self.v_proj = Linear(self.embed_dim, self.embed_dim)
    self.q_proj = Linear(self.embed_dim, self.embed_dim)
    self.out_proj = Linear(self.embed_dim, self.embed_dim)

  def _shape(self, tensor, seq_len: int, bsz: int):
    return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)

  def __call__(self, hidden_states, causal_attention_mask):
    bsz, tgt_len, embed_dim = hidden_states.shape

    query_states = self.q_proj(hidden_states) * self.scale
    key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).reshape(*proj_shape)
    key_states = key_states.reshape(*proj_shape)
    src_len = key_states.shape[1]
    value_states = value_states.reshape(*proj_shape)

    attn_weights = query_states @ key_states.permute(0,2,1)

    attn_weights = attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
    attn_weights = attn_weights.reshape(bsz * self.num_heads, tgt_len, src_len)
    
    attn_weights = attn_weights.softmax()

    attn_output = attn_weights @ value_states

    attn_output = attn_output.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.permute(0,2,1,3)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)
    return attn_output

class CLIPEncoderLayer:
  def __init__(self):
    self.self_attn = CLIPAttention()
    self.layer_norm1 = Normalize(768, num_groups=None)
    self.mlp = CLIPMLP()
    self.layer_norm2 = Normalize(768, num_groups=None)

  def __call__(self, hidden_states, causal_attention_mask):
    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)
    hidden_states = self.self_attn(hidden_states, causal_attention_mask)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states

class CLIPEncoder:
  def __init__(self):
    self.layers = [CLIPEncoderLayer() for i in range(12)]
  
  def __call__(self, hidden_states, causal_attention_mask):
    for i,l in enumerate(self.layers):
      hidden_states = l(hidden_states, causal_attention_mask)
    return hidden_states

class CLIPTextEmbeddings:
  def __init__(self):
    self.position_ids = Tensor.empty(1, 77, device="CPU")  # what is this?
    self.token_embedding = {"weight": Tensor.empty(49408, 768, device="CPU")}
    self.position_embedding = {"weight": Tensor.empty(77, 768, device="CPU")}

  def __call__(self, input_ids, position_ids):
    # TODO: actually support batches
    inputs = np.zeros((1, len(input_ids), 49408))
    positions = np.zeros((1, len(position_ids), 77))
    for i,x in enumerate(input_ids): inputs[0][i][x] = 1
    for i,x in enumerate(position_ids): positions[0][i][x] = 1

    inputs_embeds = Tensor(inputs, device=self.token_embedding['weight'].device) @ self.token_embedding['weight']
    position_embeddings = Tensor(positions, device=self.position_embedding['weight'].device) @ self.position_embedding['weight'] 

    return inputs_embeds + position_embeddings

class CLIPTextTransformer:
  def __init__(self):
    self.embeddings = CLIPTextEmbeddings()
    self.encoder = CLIPEncoder()
    self.final_layer_norm = Normalize(768, num_groups=None)

  def __call__(self, input_ids):
    x = self.embeddings(input_ids, list(range(len(input_ids))))
    causal_attention_mask = np.triu(np.ones((1,1,77,77), dtype=np.float32) * -np.inf, k=1)
    x = self.encoder(x, Tensor(causal_attention_mask, device=x.device))
    return self.final_layer_norm(x)

# Clip tokenizer, taken from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py (MIT license)
@lru_cache()
def default_bpe():
  return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")

def get_pairs(word):
  """Return set of symbol pairs in a word.
  Word is represented as tuple of symbols (symbols being variable-length strings).
  """
  pairs = set()
  prev_char = word[0]
  for char in word[1:]:
    pairs.add((prev_char, char))
    prev_char = char
  return pairs

def whitespace_clean(text):
  text = re.sub(r'\s+', ' ', text)
  text = text.strip()
  return text

def bytes_to_unicode():
  """
  Returns list of utf-8 byte and a corresponding list of unicode strings.
  The reversible bpe codes work on unicode strings.
  This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
  When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
  This is a signficant percentage of your normal, say, 32K bpe vocab.
  To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
  And avoids mapping to whitespace/control characters the bpe code barfs on.
  """
  bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
  cs = bs[:]
  n = 0
  for b in range(2**8):
    if b not in bs:
      bs.append(b)
      cs.append(2**8+n)
      n += 1
  cs = [chr(n) for n in cs]
  return dict(zip(bs, cs))

class ClipTokenizer:
  def __init__(self, bpe_path: str = default_bpe()):
    self.byte_encoder = bytes_to_unicode()
    merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
    merges = merges[1:49152-256-2+1]
    merges = [tuple(merge.split()) for merge in merges]
    vocab = list(bytes_to_unicode().values())
    vocab = vocab + [v+'</w>' for v in vocab]
    for merge in merges:
      vocab.append(''.join(merge))
    vocab.extend(['<|startoftext|>', '<|endoftext|>'])
    self.encoder = dict(zip(vocab, range(len(vocab))))
    self.bpe_ranks = dict(zip(merges, range(len(merges))))
    self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
    self.pat = self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\s]+""", re.IGNORECASE)

  def bpe(self, token):
    if token in self.cache:
      return self.cache[token]
    word = tuple(token[:-1]) + ( token[-1] + '</w>',)
    pairs = get_pairs(word)

    if not pairs:
      return token+'</w>'

    while True:
      bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
      if bigram not in self.bpe_ranks:
        break
      first, second = bigram
      new_word = []
      i = 0
      while i < len(word):
        try:
          j = word.index(first, i)
          new_word.extend(word[i:j])
          i = j
        except Exception:
          new_word.extend(word[i:])
          break

        if word[i] == first and i < len(word)-1 and word[i+1] == second:
          new_word.append(first+second)
          i += 2
        else:
          new_word.append(word[i])
          i += 1
      new_word = tuple(new_word)
      word = new_word
      if len(word) == 1:
        break
      else:
        pairs = get_pairs(word)
    word = ' '.join(word)
    self.cache[token] = word
    return word

  def encode(self, text):
    bpe_tokens = []
    text = whitespace_clean(text.strip()).lower()
    for token in re.findall(self.pat, text):
      token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
      bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
    # Truncation, keeping two slots for start and end tokens.
    if len(bpe_tokens) > 75:
      bpe_tokens = bpe_tokens[:75]
    return [49406] + bpe_tokens + [49407] * (77 - len(bpe_tokens) - 1)

class StableDiffusion:
  def __init__(self):
    self.alphas_cumprod = Tensor.empty(1000)
    self.model = namedtuple("DiffusionModel", ["diffusion_model"])(diffusion_model = UNetModel())
    self.first_stage_model = AutoencoderKL()
    self.cond_stage_model = namedtuple("CondStageModel", ["transformer"])(transformer = namedtuple("Transformer", ["text_model"])(text_model = CLIPTextTransformer()))

  # TODO: make __call__ run the model

# ** ldm.models.autoencoder.AutoencoderKL (done!)
# 3x512x512 <--> 4x64x64 (16384)
# decode torch.Size([1, 4, 64, 64]) torch.Size([1, 3, 512, 512])
# section 4.3 of paper
# first_stage_model.encoder, first_stage_model.decoder

# ** ldm.modules.diffusionmodules.openaimodel.UNetModel
# this is what runs each time to sample. is this the LDM?
# input:  4x64x64
# output: 4x64x64
# model.diffusion_model
# it has attention?

# ** ldm.modules.encoders.modules.FrozenCLIPEmbedder
# cond_stage_model.transformer.text_model

# this is sd-v1-4.ckpt
#FILENAME = "/Users/kafka/fun/mps/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt"
#FILENAME = "/home/kafka/model.ckpt"
FILENAME = "sd-v1-4.ckpt"

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run Stable Diffusion', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--steps', type=int, default=5, help="Number of steps in diffusion")
  parser.add_argument('--phrase', type=str, default="a horse sized cat eating a bagel", help="Phrase to render")
  parser.add_argument('--out', type=str, default="/tmp/rendered.png", help="Output filename")
  args = parser.parse_args()

  # WTF!! no_grad breaks it (only with OPENCL, now fixed)
  Tensor.no_grad = True
  model = StableDiffusion()

  # load in weights
  dat = fake_torch_load_zipped(open(FILENAME, "rb"))
  for k,v in tqdm(dat['state_dict'].items()):
    try:
      w = get_child(model, k)
    except (AttributeError, KeyError, IndexError):
      #traceback.print_exc()
      w = None 
    #print(f"{str(v.shape):30s}", w.shape if w is not None else w, k)
    if w is not None:
      assert w.shape == v.shape
      w.assign(v.astype(np.float32))

  # run through CLIP to get context

  tokenizer = ClipTokenizer()
  phrase = tokenizer.encode(args.phrase)
  context = model.cond_stage_model.transformer.text_model(phrase).realize()
  print("got CLIP context", context.shape)

  phrase = tokenizer.encode("")
  unconditional_context = model.cond_stage_model.transformer.text_model(phrase).realize()
  print("got unconditional CLIP context", unconditional_context.shape)

  # done with clip model
  del model.cond_stage_model

  def get_model_output(latent, t):
    # put into diffuser
    timesteps = Tensor([t])
    unconditional_latent = model.model.diffusion_model(latent, timesteps, unconditional_context).realize()
    latent = model.model.diffusion_model(latent, timesteps, context).realize()

    unconditional_guidance_scale = 7.5
    e_t = unconditional_latent + unconditional_guidance_scale * (latent - unconditional_latent)
    return e_t

  timesteps = list(np.arange(1, 1000, 1000//args.steps))
  print(f"running for {timesteps} timesteps")
  alphas = [model.alphas_cumprod.numpy()[t] for t in timesteps]
  alphas_prev = [1.0] + alphas[:-1]

  def get_x_prev_and_pred_x0(x, e_t, index):
    temperature = 1
    a_t, a_prev = alphas[index], alphas_prev[index]
    sigma_t = 0
    sqrt_one_minus_at = math.sqrt(1-a_t)
    #print(a_t, a_prev, sigma_t, sqrt_one_minus_at)

    pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)

    # direction pointing to x_t
    dir_xt = math.sqrt(1. - a_prev - sigma_t**2) * e_t
    noise = sigma_t * Tensor.randn(*x.shape) * temperature

    x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt #+ noise
    return x_prev, pred_x0

  # start with random noise
  latent = Tensor.randn(1,4,64,64)

  # this is diffusion
  for index, timestep in (t:=tqdm(list(enumerate(timesteps))[::-1])):
    t.set_description("%3d %3d" % (index, timestep))
    e_t = get_model_output(latent, timestep)
    x_prev, pred_x0 = get_x_prev_and_pred_x0(latent, e_t, index)
    #e_t_next = get_model_output(x_prev)
    #e_t_prime = (e_t + e_t_next) / 2
    #x_prev, pred_x0 = get_x_prev_and_pred_x0(latent, e_t_prime, index)
    latent = x_prev
    latent.realize()

  # upsample latent space to image with autoencoder
  x = model.first_stage_model.post_quant_conv(1/0.18215 * latent)
  x = model.first_stage_model.decoder(x)

  # make image correct size and scale
  x = (x + 1.0) / 2.0
  x = x.reshape(3,512,512).permute(1,2,0)
  dat = (x.detach().numpy().clip(0, 1)*255).astype(np.uint8)
  print(dat.shape)

  # save image
  im = Image.fromarray(dat)
  print(f"saving {args.out}")
  im.save(args.out)
