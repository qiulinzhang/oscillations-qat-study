#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch

from quantization.hijacker import QuantizationHijacker


def add_oscillation_trackers(model, max_bits=4, *args, **kwarks):
    tracker_dict = {}
    # Add oscillation trackers to all weight quantizers
    for name, module in model.named_modules():
        if isinstance(module, QuantizationHijacker):
            q = module.weight_quantizer.quantizer
            if q.n_bits > max_bits:
                print(
                    f"Skip tracking/freezing for {name}, too high bit {q.n_bits} (max {max_bits})"
                )
                continue
            int_fwd_wrapper = TrackOscillation(int_fwd=q.to_integer_forward, *args, **kwarks)
            q.to_integer_forward = int_fwd_wrapper
            tracker_dict[name + ".weight_quantizer"] = int_fwd_wrapper
    return tracker_dict


class TrackOscillation:
    """
    This is a wrapper of the int_forward function of a quantizer.
    It tracks the oscillations in integer domain.
    """

    def __init__(self, int_fwd, momentum=0.01, freeze_threshold=0, use_ema_x_int=True):
        self.int_fwd = int_fwd
        self.momentum = momentum

        self.prev_x_int = None
        self.prev_switch_dir = None

        # Statistics to log
        self.ema_oscillation = None
        self.oscillated_sum = None
        self.total_oscillation = None
        self.iters_since_reset = 0

        # Extra variables for weight freezing
        self.freeze_threshold = freeze_threshold  # This should be at least 2-3x the momentum value. 权重发生震荡后将其冻结的阈值，震荡频率
        self.use_ema_x_int = use_ema_x_int # 权重冻结后是否采用滑动平均后的值
        self.frozen = None # 在整个权重矩阵上针对每个元素是否冻结的标志位
        self.frozen_x_int = None # 冻结后的 x_int
        self.ema_x_int = None # 采用EMA滑动平均后的值

    def __call__(self, x_float, skip_tracking=False, *args, **kwargs):
        x_int = self.int_fwd(x_float, *args, **kwargs) # 浮点权重经过量化后的整型值

        # Apply weight freezing
        if self.frozen is not None: # 针对冻结矩阵将 x_int 的一部分填充为冻结后的x_int,即 self.frozen_x_int
            x_int = ~self.frozen * x_int + self.frozen * self.frozen_x_int

        if skip_tracking: # 如果不跟踪震荡频率，可以直接返回
            return x_int

        # 跟踪权重矩阵的每个元素的震荡频率
        with torch.no_grad():
            # Check if everything is correctly initialized, otherwise do so
            self.check_init(x_int) # 对一些跟踪震荡频率所需要的变量进行初始化，
            # 比如，上一次值的变化方向是否发生改变，self.prev_switch_dir
            # 上一次的整型值，self.prev_x_int

            # detect difference in x_int  NB we round to avoid int inaccuracies
            delta_x_int = torch.round(self.prev_x_int - x_int).detach()  # should be {-1, 0, 1} # 获得上一次整型值与这一次整型值之间的差距
            switch_dir = torch.sign(delta_x_int)  # This is {-1, 0, 1} as sign(0) is mapped to 0 # 根据差距的符号来判断每个元素是增大还是减小，获取每个值变化的方向
            # binary mask for switching
            switched = delta_x_int != 0 # 统计当前值发生变化的元素

            oscillated = (self.prev_switch_dir * switch_dir) == -1 # 发生震荡的条件就是 上一次变化的方向 与这一次变化的方向 相反，即其乘积为 -1
            self.ema_oscillation = ( # 用滑动平均统计每个元素发生震荡的频率 
                self.momentum * oscillated + (1 - self.momentum) * self.ema_oscillation
            )

            # Update prev_switch_dir for the switch variables
            self.prev_switch_dir[switched] = switch_dir[switched] # 将当前值发生变化的元素更新到prev,这里采用 switched 标志位的原因是有些元素可能上一次发生变化，但是当前这一次不发生变化
            self.prev_x_int = x_int
            self.oscillated_sum = oscillated.sum() # 统计发生震荡的元素个数
            self.total_oscillation += oscillated
            self.iters_since_reset += 1

            # Freeze some weights
            if self.freeze_threshold > 0: # 如果设置的震荡阈值>0
                freeze_weights = self.ema_oscillation > self.freeze_threshold # 记录下那些震荡频率大于阈值的元素
                self.frozen[freeze_weights] = True  # Set them to frozen，并把这些震荡频率大于阈值的元素位置的标志位设置为True
                if self.use_ema_x_int: # 如果采用滑动平均来记录冻结的值
                    self.frozen_x_int[freeze_weights] = torch.round(self.ema_x_int[freeze_weights]) # 那么冻结的权重就等于滑动平均的值取round
                    # Update x_int EMA which can be used for freezing
                    self.ema_x_int = self.momentum * x_int + (1 - self.momentum) * self.ema_x_int # 通过EMA统计 x_int 在哪个位置停留时间更长
                else: # 如果不采用EMA来统计
                    self.frozen_x_int[freeze_weights] = x_int[freeze_weights] # 那么就把那些需要freeze的x_int 赋值给 self.frozen_x_int

        return x_int # 

    def check_init(self, x_int):
        if self.prev_x_int is None:
            # Init prev switch dir to 0
            self.prev_switch_dir = torch.zeros_like(x_int) # 权重矩阵中，上一次每个元素的值是否发生变化
            self.prev_x_int = x_int.detach()  # Not sure if needed, don't think so
            self.ema_oscillation = torch.zeros_like(x_int) # 权重矩阵中 每个元素的震荡频率的滑动平均值
            self.oscillated_sum = 0 # 统计发生震荡的元素总个数
            self.total_oscillation = torch.zeros_like(x_int) # 统计权重矩阵中每个元素发生的震荡次数
            print("Init tracking", x_int.shape)
        else:
            assert (
                self.prev_x_int.shape == x_int.shape
            ), "Tracking shape does not match current tensor shape."

        # For weight freezing
        if self.frozen is None and self.freeze_threshold > 0:
            self.frozen = torch.zeros_like(x_int, dtype=torch.bool) # 权重矩阵中每个元素是否发生震荡
            self.frozen_x_int = torch.zeros_like(x_int) # 
            if self.use_ema_x_int:
                self.ema_x_int = x_int.detach().clone()
            print("Init freezing", x_int.shape)
