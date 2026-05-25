import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import math
import torch
import torch.nn.functional as F

debug = True

class Bank():
    def __init__(self, args):
        self.DRAM_column = args.DRAM_column
        self.DRAM_row = args.DRAM_row
        self.burst_length = args.burst_length
        self.arrays = 0 if args.only_trace else torch.zeros(torch.Size([self.DRAM_row, self.DRAM_column]))
        self.latch = 0 if args.only_trace else [0 for _ in range(args.reuse_size)]
        self.activation_function_register = 0

class Channel(Bank):
    def __init__(self, args):
        super().__init__(args)
        self.num_banks = args.num_banks
        self.GB = torch.zeros(torch.Size([self.DRAM_column]))
        bank_lst = ["bank_" + str(i) for i in range(self.num_banks)]
        self.channel = {}
        for bank in bank_lst:
            self.channel[bank] = Bank(args)

class DIMM(Channel):
    """
    DIMM Class inherits DRAM topology from Channel and Bank class
    """
    def __init__(self, args):
        super().__init__(args)
        self.num_channels = args.num_channels
        channel_lst = ["channel_" + str(i) for i in range(self.num_channels)]
        self.dimm = {}
        for channel in channel_lst:
            self.dimm[channel] = Channel(args)

class PIM():
    """
    TransformerBlock Class inherits computate functionality from PIM class
    """
    def __init__(self, args):
        self.DRAM_column = args.DRAM_column
        self.DRAM_row = args.DRAM_row
        self.burst_length = args.burst_length
        self.num_banks = args.num_banks
        self.num_channels = args.num_channels
        self.threads = args.threads
        self.pim_device = {}
        if not args.only_trace:
            if args.model_parallel:
                for i in range(args.FC_devices):
                    self.pim_device["dimm_{}".format(i)] = DIMM(args)
            else:
                self.pim_device["dimm_0"] = DIMM(args)
        self.op_trace = args.op_trace
        self.trace_file = args.trace_file
        self.file = open(self.trace_file, "w")
        # print(torch.linspace(-10, 10, 512))
        self.sigmoid_LUT = torch.sigmoid(torch.linspace(-10, 10, 512))
        self.sigmoid_LUT = torch.cat((self.sigmoid_LUT, torch.tensor([1])))
        # print(self.sigmoid_LUT)
        self.time = {
            "COPY_GB_BK": 0,
            "COPY_BK_GB": 0,
            "WR_GB": 0,
            "MAC_ABK": 0,
            "MAC_BK_BK": 0,
            "MAC_BK_GB": 0,
            "EWMUL": 0,
            "EWADD": 0,
            "AF": 0,
            "RD_MAC": 0,
            "RD_AF": 0,
            "WR_BIAS": 0,
            "RD_SBK": 0,
            "WR_SBK": 0,
            "breakdown_sa_pow": 0,
            "breakdown_sa_weight": 0,
            "breakdown_sa_score": 0,
            "breakdown_sa_output": 0,
            "breakdown_ffn_weight": 0,
            "breakdown_embedding_weight": 0,
        }
        self.timing_constant = {
            "COPY_GB_BK": 45.5,
            "COPY_BK_GB": 42.5,
            "WR_GB": 32,
            "MAC_ABK": 49,
            "MAC_BK_BK": 49,
            "MAC_BK_GB": 49,
            "EWMUL": 47,
            "EWADD": 0,
            "AF": 60,
            "RD_MAC": 37.5,
            "RD_AF": 37.5,
            "WR_BIAS": 37.5,
            "RD_SBK": 30.5,
            "WR_SBK": 45.5,
        }

    def hex_channel_mask(self, channel):
        mask = ["0"] * self.num_channels
        if isinstance(channel, list):
            for c in channel:
                mask[c] = "1"
        else:
            mask[channel] = "1"
        binary = "0b" + ''.join(mask)
        num = int(binary, 2)
        
        # convert int to hexadecimal
        hex_num = hex(num)
        return hex_num

    def address(self, dimm_index, channel_index, bank_index, row_index, col):
        bank_size = self.DRAM_column * self.DRAM_row
        channel_size = bank_size * self.num_banks
        dimm_size = channel_size * self.num_channels
        addr = dimm_index * dimm_size + channel_index * channel_size + bank_index * bank_size + row_index * self.DRAM_column + col
        return addr
    
    def W_MEM_only_trace(self, channel_index, bank_index, row_index, size):
        for i in range((size - 1) // self.burst_length + 1):
            self.file.write("W MEM {} {} {}\n".format(channel_index, bank_index, row_index))
    
    def R_MEM_only_trace(self, channel_index, bank_index, row_index, size):
        for i in range((size - 1) // self.burst_length + 1):
            self.file.write("R MEM {} {} {}\n".format(channel_index, bank_index, row_index))

    def EWMUL_only_trace(self, channel, row_index, op_size):
        # parallel in 4 bank groups, src bank 0 and 1, dest bank 2
        self.time["EWMUL"] += self.timing_constant["EWMUL"] + op_size
        self.file.write("AiM EWMUL {} {} {}\n".format(op_size, self.hex_channel_mask(channel), row_index))
    
    def COPY_BK_GB_only_trace(self, channel, bank, row_index, op_size):
        assert bank < self.num_banks
        self.time["COPY_BK_GB"] += self.timing_constant["COPY_BK_GB"] + op_size
        self.file.write("AiM COPY_BKGB {} {} {} {}\n".format(op_size, self.hex_channel_mask(channel), bank, row_index))

    def COPY_GB_BK_only_trace(self, channel, bank, row_index, op_size):
        assert bank < self.num_banks
        self.time["COPY_GB_BK"] += self.timing_constant["COPY_GB_BK"] + op_size
        self.file.write("AiM COPY_GBBK {} {} {} {}\n".format(op_size, self.hex_channel_mask(channel), bank, row_index))

    def SYNC_only_trace(self):
        self.file.write("AiM SYNC\n")
    
    def finish(self):
        self.file.write("AiM EOC\n")

    def WR_SBK_only_trace(self, channel, bank, row_index, op_size): #MAINUDDIN
        #assert bank < self.num_banks
        self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + op_size
        self.file.write("W MEM {} {} {} {}\n".format(op_size, self.hex_channel_mask(channel), bank, row_index))

    def RD_SBK_only_trace(self, channel, bank, row_index, op_size):
        #assert bank < self.num_banks
        self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + op_size
        self.file.write("R MEM {} {} {} {}\n".format(op_size, self.hex_channel_mask(channel), bank, row_index))

class TransformerBlock(PIM):
    """
    Llama TransformerBlock Class inherits computate functionality from PIM class
    """
    def __init__(self, dic_model, args):
        super().__init__(args)
        self.pim_compute = args.pim_compute
        if args.op_trace:
            self.trace_prepare = True
            self.trace_norm = True
            self.trace_fc_kqvo = True
            self.trace_attention = True
            self.trace_softmax = True
            self.trace_fc_kqvo = True
            self.trace_fc_ffn = True
            self.trace_activation = True
        else:
            self.trace_prepare = args.trace_prepare
            self.trace_norm = args.trace_norm
            self.trace_fc_kqvo = args.trace_fc_kqvo
            self.trace_attention = args.trace_attention
            self.trace_softmax = args.trace_softmax
            self.trace_fc_kqvo = args.trace_fc_kqvo
            self.trace_fc_ffn = args.trace_fc_ffn
            self.trace_activation = args.trace_activation
        self.model = args.model
        self.seqlen = args.seqlen
        self.vocab_size = 32000
        self.FC_devices = args.FC_devices
        self.embedding = args.embedding
        self.only_FC = args.only_FC
        self.only_trace = args.only_trace
        self.model_parallel = args.model_parallel
        self.pipeline_parallel = args.pipeline_parallel
        if args.channels_per_block:
            self.channels_per_block = args.channels_per_block
        else:
            self.channels_per_block = args.num_channels
        self.GEMV_order = args.GEMV
        self.reuse_size = args.reuse_size
        if "TP_param" in dic_model.keys():
            self.TP_param = dic_model["TP_param"].item()
        else:
            self.TP_param = 1
        self.dim = dic_model["dim"].item()
        self.n_heads = dic_model["n_heads"].item()
        self.head_dim = self.dim // self.n_heads // self.TP_param
        self.max_seq_len = args.max_seq_len
        self.GQA = False
        self.inter_device_attention = args.inter_device_attention
        self.n_repeat = 1
        if "n_kv_heads" in dic_model.keys():
            self.GQA = True
            self.n_kv_heads = dic_model["n_kv_heads"].item()
            self.n_repeat = self.n_heads // self.n_kv_heads
        else:
            self.n_kv_heads = self.n_heads
        self.x = dic_model["x"].float()
        self.SANorm = dic_model["SANorm"].float()
        self.FFNNorm = dic_model["FFNNorm"].float()
        if "freqs_cis" in dic_model.keys():
            self.freqs_cis = dic_model["freqs_cis"]
        self.start_pos = dic_model["start_pos"]
        self.sa = dic_model["sa"].float()
        self.h = dic_model["h"].float()
        self.out = dic_model["out"].float()
        self.wq = dic_model["wq"].float()
        self.wk = dic_model["wk"].float()
        self.wv = dic_model["wv"].float()
        self.xq = dic_model["xq"].float()
        self.xk = dic_model["xk"].float()
        self.xv = dic_model["xv"].float()
        self.cache_k = dic_model["cache_k"].float()
        self.cache_v = dic_model["cache_v"].float()
        self.scores = dic_model["scores"].float()
        self.output = dic_model["output"].float()
        self.wo = dic_model["wo"].float()
        self.w1 = dic_model["w1"].float()
        self.w2 = dic_model["w2"].float()
        if "w3" in dic_model.keys():
            self.w3 = dic_model["w3"].float()
        self.ffn = dic_model["ffn"].float()
        self.mode = {"vector":0, "weights":1, "cache_k":2, "cache_v":3, "score":4}
        self.total_banks = self.channels_per_block * self.num_banks
        if self.model_parallel:
            self.FC_total_banks = self.total_banks * self.FC_devices
            self.intra_device_attention = False if self.inter_device_attention else True
            banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1
            if banks_per_head < self.num_banks:
                self.intra_device_attention = True
        else:
            self.FC_total_banks = self.total_banks
            self.intra_device_attention = True

    def bank_index(self, index):
        # look for the bank to store a head
        dimm_index = index // (self.num_banks * self.num_channels)
        channel_index = (index - dimm_index * self.num_banks * self.num_channels) // self.num_banks
        bank_index = index % self.num_banks
        return dimm_index, channel_index, bank_index

    def store_for_EWMUL_input_only_trace(self, channels_required, total_banks, bank_group_index, row_index, size):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        for i in range((size - 1) // self.burst_length + 1):
            for bank in range(total_banks):
                dimm_index, channel_index, bank_index = self.bank_index(bank*4+bank_group_index)
                for tb in range(num_transformer_blocks_per_device):
                    self.W_MEM_only_trace(channel_index + channels_required * tb, bank_index, row_index, self.burst_length)
    
    def load_from_EWMUL_input_only_trace(self, channels_required, total_banks, bank_group_index, row_index, size):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        for i in range((size - 1) // self.burst_length + 1):
            for bank in range(total_banks):
                dimm_index, channel_index, bank_index = self.bank_index(bank*4+bank_group_index)
                for tb in range(num_transformer_blocks_per_device):
                    self.R_MEM_only_trace(channel_index + channels_required * tb, bank_index, row_index, self.burst_length)

    def COPYBK_EWMUL_input_only_trace(self, channels_required, input_vector_EWMUL_utilized_banks, bank_group_index, row_index, size):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        utilized_banks_per_channel=input_vector_EWMUL_utilized_banks//channels_required #for channel parallelism
        opsize= (size - 1) // self.burst_length + 1 #opsize is sent, not size
        for bank in range(utilized_banks_per_channel):
            for tb in range(num_transformer_blocks_per_device):
                channel_lst=list( range(channels_required*tb, channels_required*(tb+1)) )
                self.COPY_GB_BK_only_trace(channel_lst, bank+bank_group_index, row_index, opsize) #channel list is sent, not number of channels

    def COPYGB_EWMUL_input_only_trace(self, channels_required, input_vector_EWMUL_utilized_banks, bank_group_index, row_index, size):
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        utilized_banks_per_channel=input_vector_EWMUL_utilized_banks//channels_required #for channel parallelism
        opsize= (size - 1) // self.burst_length + 1 #opsize is sent, not size
        for bank in range(utilized_banks_per_channel):
            for tb in range(num_transformer_blocks_per_device):
                channel_lst_multi=list( range(channels_required*tb, channels_required*(tb+1)) )
                self.COPY_BK_GB_only_trace(channel_lst_multi, bank+bank_group_index, row_index, opsize)  #channel list is sent, not number of channels
                