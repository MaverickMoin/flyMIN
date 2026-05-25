import math
import torch
import torch.nn.functional as F
from new_sim import PIM, TransformerBlock

debug = True

class TransformerBlockLlama(TransformerBlock):
    """
    TransformerBlock Class inherits computate functionality from PIM class
    """
    def __init__(self, dic_model, args):
        super().__init__(dic_model, args)
        
        self.rd_q_burst=0;
        self.store_for_rd_q_burst=0;
        self.rd_k_burst =0;
        self.store_for_rd_k_burst=0;
        
        self.inv_q_burst=0;
        self.store_for_inv_q_burst=0;
        self.inv_k_burst=0;
        self.store_for_inv_k_burst=0;

        self.ewmul_q_burst=0;
        self.ewmul_k_burst=0;


    def memory_mapping(self):
        """
        Each chip has 1GB density (2 channels) w/ 32 banks (16 banks/channel).
        Each bank has 32MB and could store 16M BF16 values (2B), i.e., 16k rows and 1k columns.

        LLaMA-2-7B
        input: (1 x 4k) stored in 16 banks (256, 1 row) and broadcast to 2 channels
        wq wk wv: 32 heads (4k x 128, 1.5k row) stored in 2 channels, 32 banks
        xq xk xv: 32 heads (1 x 128, 3 row) stored in 2 channels, 32 banks
        cache_k: 32 heads (128 x L, 0.5k row) stored in 2 channels, 32 banks
        scores: 32 heads (1 x L, 4 row) stored in 2 channels, 32 banks
        cache_v: 32 heads (L x 128, 0.5k row) stored in 2 channels, 32 banks
        output: 32 heads (1 x 128, 1 row) stored in 2 channels, 32 banks
        wo: 32 heads (4k x 128, 0.5k row) stored in 2 channels, 32 banks
        sa: (1 x 4k) stored in 16 banks (256, 1 row) and broadcast to 2 channels
        w1 w3: 32 parts (4k x 344, 2*1376 rows) stored in 2 channels, 32 banks
        x1 x3: 32 parts (1 x 344, 2 rows) stored in 2 channels, 32 banks
        x1 sigmoid: 32 parts (1 x 344, 1 rows) stored in 2 channels, 32 banks
        x3: (1 x 11008) stored in 16 banks (688, 1 row) and broadcast to 2 channels
        w2: 32 parts (11008 x 128, 1376 rows) stored in 2 channels, 32 banks
        out: 32 parts (1 x 128, 1 rows) stored in 2 channels, 32 banks
        SiLU AF: TODO
        """
        self.dic_size = {}
        self.dic_row = {}
        self.dic_shape = {}
        self.dic_size["x"] = self.x.reshape(-1).shape[0]
        self.dic_row["x"] = (self.dic_size["x"] // self.num_banks - 1) // self.DRAM_column + 1
        total_banks = self.total_banks
        if self.model_parallel:
            FC_total_banks = total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = total_banks
            channels_required = self.channels_per_block
        assert self.dic_size["x"] == self.dim
        # print("x\t\t\t {} x {}\t\t\t requires {} rows".format(1, self.dic_size["x"], self.dic_row["x"]))
        # print("x_copy\t\t {} x {}\t\t\t requires {} rows".format(1, self.dic_size["x"], self.dic_row["x"]))
        # print("SANorm\t\t {} x {}\t\t\t requires {} rows".format(1, self.dic_size["x"], self.dic_row["x"]))
        # print("FFNNorm\t\t {} x {}\t\t\t requires {} rows".format(1, self.dic_size["x"], self.dic_row["x"]))

        self.dic_size["wq"] = self.wq.reshape(-1).shape[0]
        assert self.dic_size["wq"] == self.n_heads * self.head_dim * self.dim
        self.dic_row["wq"] = ((self.wq.shape[0] - 1) // FC_total_banks + 1) * ((self.wq.shape[1] - 1) // self.DRAM_column + 1)
        # print("wq\t\t\t {} x {} x {}\t requires {} rows".format(self.n_heads, self.dim, self.head_dim, self.dic_row["wq"]))
        self.dic_size["wk"] = self.wk.reshape(-1).shape[0]

        assert self.dic_size["wk"] == self.n_kv_heads * self.head_dim * self.dim
        self.dic_row["wk"] = ((self.wk.shape[0] - 1) // FC_total_banks + 1) * ((self.wk.shape[1] - 1) // self.DRAM_column + 1)
        # print("wk\t\t\t {} x {} x {} \t requires {} rows".format(self.n_heads, self.dim, self.head_dim, self.dic_row["wk"]))
        self.dic_size["wv"] = self.wv.reshape(-1).shape[0]
        assert self.dic_size["wv"] == self.n_kv_heads * self.head_dim * self.dim
        self.dic_row["wv"] = ((self.wv.shape[0] - 1) // FC_total_banks + 1) * ((self.wv.shape[1] - 1) // self.DRAM_column + 1)
        # print("wv\t\t\t {} x {} x {} \t requires {} rows".format(self.n_heads, self.dim, self.head_dim, self.dic_row["wv"]))

        self.dic_row["xq"] = 1
        self.dic_row["xk"] = 1
        self.dic_row["xv"] = 1

        self.dic_size["cache_k"] = self.max_seq_len * self.n_kv_heads * self.head_dim
        assert self.cache_k.reshape(-1).shape[0] == (self.start_pos + 1) * self.n_kv_heads * self.head_dim
        self.dic_row["cache_k"] = ((self.max_seq_len - 1) // self.FC_total_banks + 1) * ((self.n_kv_heads * self.head_dim - 1) // self.DRAM_column + 1)
        # print("cache_k\t\t {} x {} x {}\t\t requires {} rows".format(self.n_kv_heads, self.head_dim, "L", self.dic_row["cache_k"]))

        self.dic_size["scores"] = self.max_seq_len * self.n_kv_heads
        assert self.scores.reshape(-1).shape[0] == (self.start_pos + 1) * self.n_heads
        num_heads_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
        self.dic_row["scores"] = ((self.max_seq_len - 1) // self.DRAM_column + 1) * num_heads_per_bank
        # print("scores\t\t {} x {} x {}\t\t\t requires {} rows".format(self.n_heads, 1, "L", self.dic_row["scores"]))

        self.dic_size["cache_v"] = self.max_seq_len * self.n_kv_heads * self.head_dim
        assert self.cache_v.reshape(-1).shape[0] == (self.start_pos + 1) * self.n_kv_heads * self.head_dim
        if self.intra_device_attention:
            self.dic_row["cache_v"] = ((self.max_seq_len - 1) // self.DRAM_column + 1) * ((self.n_kv_heads - 1) // self.channels_per_block + 1) * ((self.head_dim - 1) // self.num_banks + 1)
        else:
            num_banks_per_head = (FC_total_banks - 1) // self.n_kv_heads + 1
            self.dic_row["cache_v"] = (self.max_seq_len - 1) // (self.DRAM_column * num_banks_per_head // self.head_dim) + 1
        # print("cache_v\t\t {} x {} x {}\t\t requires {} rows".format(self.n_kv_heads, "L", self.head_dim, self.dic_row["cache_v"]))

        self.dic_size["output"] = self.output.reshape(-1).shape[0]
        assert self.dic_size["output"] == self.n_heads * self.head_dim
        self.dic_row["output"] = (self.dic_size["output"] // total_banks - 1) // self.DRAM_column + 1
        # print("output\t\t {} x {}\t\t\t requires {} rows".format(1, self.n_heads * self.head_dim, self.dic_row["output"]))

        self.dic_size["wo"] = self.wo.reshape(-1).shape[0]
        assert self.dic_size["wo"] == self.n_heads * self.head_dim * self.dim
        self.dic_row["wo"] = ((self.wo.shape[0] - 1) // FC_total_banks + 1) * ((self.wo.shape[1] - 1) // self.DRAM_column + 1)
        # print("wo\t\t\t {} x {} x {}\t requires {} rows".format(self.n_heads, self.dim, self.head_dim, self.dic_row["wo"]))
        self.dic_size["sa"] = self.sa.reshape(-1).shape[0]
        assert self.dic_size["sa"] == self.n_heads * self.head_dim
        self.dic_row["sa"] = (self.dic_size["sa"] // total_banks - 1) // self.DRAM_column + 1
        # print("sa\t\t\t {} x {}\t\t\t requires {} rows".format(1, self.n_heads * self.head_dim, self.dic_row["sa"]))

        ffn_dim = self.w1.shape[0]
        ffn_parallel_dim = (ffn_dim - 1) // total_banks + 1
        ffn_FC_dim = (ffn_dim - 1) // FC_total_banks + 1

        self.dic_size["w1"] = self.w1.reshape(-1).shape[0]
        assert self.dic_size["w1"] == ffn_dim * self.dim
        self.dic_row["w1"] = ffn_FC_dim * ((self.dim - 1) // self.DRAM_column + 1)
        # print("w1\t\t\t {} x {} x {}\t requires {} rows".format(self.n_heads, self.dim, ffn_FC_dim, self.dic_row["w1"]))
        self.dic_size["w3"] = self.w3.reshape(-1).shape[0]
        assert self.dic_size["w3"] == ffn_dim * self.dim
        self.dic_row["w3"] = ffn_FC_dim * ((self.dim - 1) // self.DRAM_column + 1)
        # print("w3\t\t\t {} x {} x {}\t requires {} rows".format(self.n_heads, self.dim, ffn_FC_dim, self.dic_row["w3"]))
        self.dic_size["x1"] = ffn_dim
        self.dic_row["x1"] = (self.dic_size["x1"] // total_banks - 1) // self.DRAM_column + 1
        # print("x1\t\t\t {} x {} x {}\t\t requires {} rows".format(self.n_heads, 1, ffn_parallel_dim, self.dic_row["x1"]))
        self.dic_size["x3"] = ffn_dim
        self.dic_row["x3"] = (self.dic_size["x3"] // total_banks - 1) // self.DRAM_column + 1
        # print("x3\t\t\t {} x {} x {}\t\t requires {} rows".format(self.n_heads, 1, ffn_parallel_dim, self.dic_row["x3"]))
        self.dic_size["x1_sigmoid"] = ffn_dim
        self.dic_row["x1_sigmoid"] = (self.dic_size["x1_sigmoid"] // total_banks - 1) // self.DRAM_column + 1
        # print("x1_sigmoid\t {} x {} x {}\t\t requires {} rows".format(self.n_heads, 1, ffn_parallel_dim, self.dic_row["x1_sigmoid"]))
        self.dic_size["w2"] = self.w2.reshape(-1).shape[0]
        assert self.dic_size["w2"] == ffn_dim * self.dim
        self.dic_row["w2"] = ((self.dim - 1) // FC_total_banks + 1) * ((ffn_dim - 1) // self.DRAM_column + 1)
        # print("w2\t\t\t {} x {} x {}\t requires {} rows".format(self.n_heads, ffn_dim, self.head_dim, self.dic_row["w2"]))
        self.dic_size["ffn"] = self.ffn.reshape(-1).shape[0]
        assert self.dic_size["ffn"] == self.dim
        self.dic_row["ffn"] = (self.dic_size["ffn"] // total_banks - 1) // self.DRAM_column + 1
        # print("ffn\t\t\t {} x {}\t\t\t requires {} rows".format(1, self.dim, self.dic_row["ffn"]))

        size = sum([self.dic_size[key] for key in self.dic_size.keys()])
        rows = sum([self.dic_row[key] for key in self.dic_row.keys()])

        DIMMs_required = (channels_required - 1) // self.num_channels + 1
        # print("\nAllocated {} DIMMs {} Channels".format(DIMMs_required, channels_required))
        # print(size * 2 // (1024 * 1024), "MB are required in {} channels".format(self.channels_per_block))
        # print(rows, "rows are required in a bank")
        task_level_parallelism = (self.DRAM_row - rows) // (self.dic_row["cache_k"] + self.dic_row["cache_v"]) + 1
        # print(task_level_parallelism, "tasks are available to execute in parallel\n")
        # dimm_lst = ["dimm_" + str(i) for i in range(DIMMs_required)]
        # self.pim_device = {}
        # for dimm in dimm_lst:
        #     self.pim_device[dimm] = DIMM(args)


        # x in neighbor bank
        self.x_row_index = 0
        # x in a bank group
        self.x_copy_row_index = self.x_row_index + self.dic_row["x"]
        # SANorm
        self.SANorm_row_index = self.x_copy_row_index + self.dic_row["x"]
        # wq
        self.wq_row_index = self.SANorm_row_index + self.dic_row["x"]
        # wk
        self.wk_row_index = self.wq_row_index + self.dic_row["wq"]
        # wv
        self.wv_row_index = self.wk_row_index + self.dic_row["wk"]
        # xq
        self.xq_row_index = self.wv_row_index + self.dic_row["wv"]
        # xk
        self.xk_row_index = self.xq_row_index + self.dic_row["xk"]
        # cache_k
        self.cache_k_row_index = self.xk_row_index + self.dic_row["xq"]
        # scores
        self.scores_row_index = self.cache_k_row_index + self.dic_row["cache_k"]
        # cache_v
        self.cache_v_row_index = self.scores_row_index + self.dic_row["scores"]
        # output
        self.output_row_index = self.cache_v_row_index + self.dic_row["cache_v"]
        # wo
        self.wo_row_index = self.output_row_index + self.dic_row["output"]
        # sa
        self.sa_row_index = self.wo_row_index + self.dic_row["wo"]
        # sa copy
        self.sa_copy_row_index = self.sa_row_index + self.dic_row["sa"]
        # FFNNorm
        self.FFNNorm_row_index = self.sa_copy_row_index + self.dic_row["sa"]
        # w1
        self.w1_row_index = self.FFNNorm_row_index + self.dic_row["sa"]
        # w3
        self.w3_row_index = self.w1_row_index + self.dic_row["w1"]
        # x1
        self.x1_row_index = self.w3_row_index + self.dic_row["w3"]
        # x3
        self.x3_row_index = self.x1_row_index + self.dic_row["x1"]
        # x1_sigmoid
        self.x1_sigmoid_row_index = self.x3_row_index + self.dic_row["x3"]
        # ffn_vector
        self.dic_row["ffn_vector"] = (self.w1.shape[0] - 1) // (self.DRAM_column * self.num_banks) + 1
        self.ffn_vector_row_index = self.x1_sigmoid_row_index + self.dic_row["x1"]
        # w2
        self.w2_row_index = self.ffn_vector_row_index + self.dic_row["ffn_vector"]
        # ffn
        self.ffn_row_index = self.w2_row_index + self.dic_row["w2"]

    def trace_only(self):
        bsz, _, _ = self.x.shape
        seqlen = self.seqlen
        total_banks = self.total_banks
        if self.model_parallel:
            FC_total_banks = total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = total_banks
            channels_required = self.channels_per_block

        channel_multi_transformer_block_required = self.num_channels // channels_required * channels_required
        channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)
        
        input_vector_EWMUL_length = (self.dim - 1) // (total_banks // 4) + 1
        input_vector_EWMUL_utilized_banks = (self.dim - 1) // input_vector_EWMUL_length + 1

        # utilized_banks_per_channel=input_vector_EWMUL_utilized_banks//channels_required
        # base_op_size=input_vector_EWMUL_length*2// self.burst_length
        # channel_lst_multi = list(range(channels_required*num_transformer_blocks_per_device))

        # K/Q/V GEMV
        if self.trace_fc_kqvo:
            # CXL Port
            # Store re-mapped xq/xk for EWMUL
            self.COPYGB_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.xq_row_index, input_vector_EWMUL_length * 2)

            self.COPYGB_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.xq_row_index, input_vector_EWMUL_length // self.n_repeat * 2)
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)
            # Rotary embedding
            self.EWMUL_only_trace(channel_lst, self.xq_row_index, self.dim // self.burst_length)
            self.EWMUL_only_trace(channel_lst, self.xk_row_index, self.dim // self.n_repeat // self.burst_length)
            # Load rotary embedding results
            self.COPYGB_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xq_row_index, input_vector_EWMUL_length * 2)

            self.COPYGB_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xq_row_index, input_vector_EWMUL_length // self.n_repeat * 2)
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)

