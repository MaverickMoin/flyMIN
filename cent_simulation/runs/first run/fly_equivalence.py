        if self.trace_fc_kqvo:
            print("============================== ROT EMB ==== {} ====================================".format(self.model))

            wr_q_burst_l=0;
            wr_k_burst_l=0;
            rd_q_burst_l=0;
            rd_k_burst_l=0;

            input_vector_EWMUL_length = (self.dim - 1) // (self.total_banks // 4) + 1
            input_vector_EWMUL_utilized_banks = (self.dim - 1) // input_vector_EWMUL_length + 1
            num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)

            utilized_banks_per_channel = (input_vector_EWMUL_utilized_banks - 1) // channels_required + 1 #for copy

            total_channels=num_transformer_blocks_per_device*channels_required #for copy
            channel_lst_multi_transformer_block = [channel for channel in range(total_channels)] #ewmul

            base_op_size = input_vector_EWMUL_length * 2 // self.burst_length

            print("Number of banks per device: ", self.num_banks)
            print("Number of blocks per device: ", num_transformer_blocks_per_device)
            print("Channels required for rotation of single block: ", channels_required)
            print("Channels required for rotation of all blocks: ", channels_required*num_transformer_blocks_per_device)
            print("All channels list for multi transformer blocks for rotation: ", channel_lst_multi_transformer_block)
            print("EWMUL length={} and utilized banks={}".format(input_vector_EWMUL_length, input_vector_EWMUL_utilized_banks  ))
            print("")
            #LOGIC: self.total_banks = self.channels_per_block * self.num_banks
            print("Total banks: {}, Channels per block: {}, Banks per device: {}".format(self.total_banks, self.channels_per_block, self.num_banks))

            print("EWMUL length={} and utilized banks={} and per channel banks={}".format(input_vector_EWMUL_length, input_vector_EWMUL_utilized_banks, utilized_banks_per_channel  ))
            print("")

            # Store re-mapped xq/xk for EWMUL
            self.COPYGB_for_EWMUL_input_only_trace(channel_lst_multi_transformer_block, utilized_banks_per_channel, 0, self.xq_row_index, base_op_size)
            #self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.rd_q_burst=total_channels*utilized_banks_per_channel*base_op_size

            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.store_for_rd_q_burst=num_transformer_blocks_per_device*input_vector_EWMUL_utilized_banks*input_vector_EWMUL_length * 2 // self.burst_length

            self.COPYGB_for_EWMUL_input_only_trace(channel_lst_multi_transformer_block, utilized_banks_per_channel, 0, self.xk_row_index, base_op_size // self.n_repeat)
            #self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.rd_k_burst=total_channels*utilized_banks_per_channel*base_op_size // self.n_repeat 

            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)
            self.store_for_rd_k_burst=num_transformer_blocks_per_device*input_vector_EWMUL_utilized_banks*input_vector_EWMUL_length // self.n_repeat * 2 // self.burst_length

            # Rotary embedding
            self.EWMUL_only_trace(channel_lst_multi_transformer_block, self.xq_row_index, self.dim // self.burst_length)
            self.ewmul_q_burst=num_transformer_blocks_per_device*self.dim // self.burst_length

            self.EWMUL_only_trace(channel_lst_multi_transformer_block, self.xk_row_index, self.dim // self.n_repeat // self.burst_length)
            self.ewmul_k_burst=num_transformer_blocks_per_device*self.dim // self.n_repeat // self.burst_length
            
            # Load rotary embedding results
            self.COPYGB_for_EWMUL_input_only_trace(channel_lst_multi_transformer_block, utilized_banks_per_channel, 2, self.xq_row_index, base_op_size)
            #self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.inv_q_burst=total_channels*utilized_banks_per_channel*base_op_size
            
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.store_for_inv_q_burst=num_transformer_blocks_per_device*input_vector_EWMUL_utilized_banks*input_vector_EWMUL_length * 2 // self.burst_length

            self.COPYGB_for_EWMUL_input_only_trace(channel_lst_multi_transformer_block, utilized_banks_per_channel, 2, self.xk_row_index, base_op_size // self.n_repeat )
            #self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.inv_k_burst=total_channels*utilized_banks_per_channel*base_op_size // self.n_repeat 
            
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)
            self.store_for_inv_k_burst=num_transformer_blocks_per_device*input_vector_EWMUL_utilized_banks*input_vector_EWMUL_length // self.n_repeat * 2 // self.burst_length
            print("dim={} and burst_length={}".format(self.dim , self.burst_length))
            print("Total elements turned number of block * dimension ={}".format(self.dim*num_transformer_blocks_per_device))
            print("Model: {}, RD-WR SBK for Q and K is same={}, ewmul_q_burst={}, ewmul_k_burst={}".format(self.model, self.dim * 2 // self.burst_length, self.dim // self.burst_length, self.dim // self.n_repeat // self.burst_length))
            print("================================ END OF ROT EMB TRACE ==================================")


        def COPYGB_for_EWMUL_input_only_trace(channel_lst_multi_transformer_block, utilized_banks_per_channel, bank_grp_index, row_index, base_op_size)
            for bank in range(utilized_banks_per_channel):
                print("CopyGB Q: Bank {}, Row index {}, Op size {}".format(bank*4+bank_grp_index, row_index, base_op_size))
                self.COPY_MIN_GB_only_trace(channel_lst_multi_transformer_block, bank*4+bank_grp_index, row_index, base_op_size)  ##INPUT Bank=  : bank_group_index=0
                wr_q_burst_l += base_op_size
                
                #print("MULINT for Q shuffle: Channel list {}, Op size {}".format(channel_lst_multi_transformer_block, op_size_xq * 2 ))
                #self.MULINT_only_trace(channel_lst_multi_transformer_block, op_size_xq * 2 )
                print("CopyGB Q: Bank {}, Total burst till now: {}".format(bank*4+bank_grp_index, wr_q_burst_l))

======================================

    def apply_rotary_emb_pim_only_trace(self, channel_lst_multi_transformer_block, channels_required, row_index_xq, row_index_xk, op_trace):
        #For Pipeline parallel, needs to have number of TB per device logic
        if not op_trace:
            return

        print("============================== ROT EMB ==== {} ====================================".format(self.model))
        channel_lst = [channel for channel in range(channels_required)]
        print("Channels required for rotation of single block: ", channels_required)
        print("Number of banks per device: ", self.num_banks)
        print("Channels list for multi transformer blocks for rotation: ", channel_lst_multi_transformer_block)

        #LOGIC: self.total_banks = self.channels_per_block * self.num_banks
        print("Total banks: {}, Channels per block: {}, Banks per device: {}".format(self.total_banks, self.channels_per_block, self.num_banks))

        # Unified Bank Utilization Math (Same for Q and K)
        input_vector_EWMUL_length = (self.dim - 1) // (self.total_banks // 4) + 1
        input_vector_EWMUL_utilized_banks = (self.dim - 1) // input_vector_EWMUL_length + 1
        utilized_banks_per_channel = (input_vector_EWMUL_utilized_banks - 1) // channels_required + 1
        print("EWMUL length={} and utilized banks={} and per channel banks={}".format(input_vector_EWMUL_length, input_vector_EWMUL_utilized_banks, utilized_banks_per_channel  ))
        print("")

        # Calculate operation sizes
        #base_op_size = self.dim // self.burst_length * input_vector_EWMUL_length 
        base_op_size = (input_vector_EWMUL_length - 1) // self.burst_length + 1 #Each bank (elements/burst_length= burst size)
        
        op_size_xq = base_op_size
        op_size_xk = base_op_size // self.n_repeat 
        #n_repeat works across the heads, meaning the embedding is common across heads
        print("Model: {}, base_op_size={}, op_size_xq={}, op_size_xk={}".format(self.model, base_op_size, op_size_xq, op_size_xk))
        wr_q_burst_l=0;
        wr_k_burst_l=0;
        rd_q_burst_l=0;
        rd_k_burst_l=0;

        # 1. Rotate the Query Vectors (xq)
        # Shuffle (Bank-by-bank locally within the channel)
        for bank in range(utilized_banks_per_channel):
            print("CopyGB Q: Bank {}, Row index {}, Op size {}".format(bank*4, row_index_xq, op_size_xq * 2 ))
            self.COPY_MIN_GB_only_trace(channel_lst_multi_transformer_block, bank*4, row_index_xq, op_size_xq * 2 )  ##INPUT Bank=  : bank_group_index=0
            wr_q_burst_l += (op_size_xq * 2 );
            
            #print("MULINT for Q shuffle: Channel list {}, Op size {}".format(channel_lst_multi_transformer_block, op_size_xq * 2 ))
            #self.MULINT_only_trace(channel_lst_multi_transformer_block, op_size_xq * 2 )
            print("CopyGB Q: Bank {}, Total burst till now: {}".format(bank*4, wr_q_burst_l))

        self.wr_q_burst=wr_q_burst_l*channel_lst_multi_transformer_block
        #NOTE: COPY_BK_GB_only_trace has assert bank < self.num_banks, to ensure number of banks are within same channel_lst_multi_transformer_block
        #Thus input_vector_EWMUL_utilized_banks cannot be used, as it is total banks needed. Not banks inside the channel.
            
        # Writeback and Multiply (Channel-wide broadcast)
        #input_vector_EWMUL_length * 2 as complex representation
        self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.xq_row_index, input_vector_EWMUL_length * 2)
        self.store_for_wr_q_burst += num_transformer_blocks_per_device* input_vector_EWMUL_utilized_banks* input_vector_EWMUL_length * 2// self.burst_length
        
        # 2. Rotate the Key Vectors (xk)
        # Shuffle (Bank-by-bank locally within the channel)
        for bank in range(utilized_banks_per_channel):
            print("CopyGB K: Bank {}, Row index {}, Op size {}".format(bank*4, row_index_xk, op_size_xk * 2 ))
            self.COPY_MIN_GB_only_trace(channel_lst_multi_transformer_block, bank*4, row_index_xk, op_size_xk * 2 )  ##INPUT Bank=  : bank_group_index=0
            wr_k_burst_l += (op_size_xk * 2 );

            #print("MULINT for K shuffle: Channel list {}, Op size {}".format(channel_lst_multi_transformer_block, op_size_xk * 2 ))  
            #self.MULINT_only_trace(channel_lst_multi_transformer_block, op_size_xk * 2 )
            print("CopyGB K: Bank {}, Total burst till now: {}".format(bank*4, wr_k_burst_l))
        self.wr_k_burst=wr_k_burst_l*channel_lst_multi_transformer_block

        # Writeback and Multiply (Channel-wide broadcast)
        self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)
        self.store_for_wr_k_burst += num_transformer_blocks_per_device* input_vector_EWMUL_utilized_banks* input_vector_EWMUL_length// self.n_repeat * 2 // self.burst_length

        #EW_MUL CHmask OPsize RO, has no BANK parameter
        print("After store for EWMUL input for Q, trace the EWMUL input channel {} ,utilized banks {} , row {},  opsize {}".format(channels_required, input_vector_EWMUL_utilized_banks, self.xq_row_index, input_vector_EWMUL_length * 2))
        # self.EWMUL_only_trace(channel_lst_multi_transformer_block, row_index_xq, op_size_xq)
        # self.ewmul_q_burst += op_size_xq;
        self.EWMUL_only_trace(channel_lst_multi_transformer_block, self.xq_row_index, self.dim // self.burst_length)
        self.ewmul_q_burst += (num_transformer_blocks_per_device* self.dim // self.burst_length)

        print("After store for EWMUL input for K, trace the EWMUL input channel {} ,utilized banks {} , row {},  opsize {}:".format(channels_required, input_vector_EWMUL_utilized_banks, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2))
        # self.EWMUL_only_trace(channel_lst_multi_transformer_block, row_index_xk, op_size_xk)
        # self.ewmul_k_burst += op_size_xk
        self.EWMUL_only_trace(channel_lst_multi_transformer_block, self.xk_row_index, self.dim // self.n_repeat // self.burst_length)
        self.ewmul_k_burst += (num_transformer_blocks_per_device* self.dim // self.n_repeat // self.burst_length)

        print("EWMUL: Q total burst {}, K total burst: {}".format(self.ewmul_q_burst, self.ewmul_k_burst))

        # 3. Inverse Shuffle (Q Vectors)
        for bank in range(utilized_banks_per_channel):
            print("CopyGB Q: Bank {}, Row index {}, Op size {}".format(bank*4+2 , row_index_xq, op_size_xq * 2 ))
            self.COPY_MIN_GB_only_trace(channel_lst_multi_transformer_block, bank*4+2, row_index_xq, op_size_xq * 2 )  ##OUTPUT Bank=  : bank_group_index=2
            rd_q_burst_l += (op_size_xq * 2 );
            
            #print("Inverse for Q shuffle: Channel list {}, Op size {}".format(channel_lst_multi_transformer_block, op_size_xq * 2 ))
            #self.MULINT_only_trace(channel_lst_multi_transformer_block, op_size_xq * 2 )
            print("CopyGB Q: Bank {}, Total burst till now: {}".format(bank*4+2, rd_q_burst_l))
        self.rd_q_burst=rd_q_burst_l*channel_lst_multi_transformer_block

        #input_vector_EWMUL_length * 2 as complex representation
        self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xq_row_index, input_vector_EWMUL_length * 2) 
        self.load_for_rd_q_burst += num_transformer_blocks_per_device* input_vector_EWMUL_utilized_banks* input_vector_EWMUL_length * 2// self.burst_length
        
        print("After LOAD for EWMUL input for Q, trace the EWMUL input channel {} ,utilized banks {} , row {},  opsize {}".format(channels_required, input_vector_EWMUL_utilized_banks, self.xq_row_index, input_vector_EWMUL_length * 2))

        # 4. Inverse Shuffle (K Vectors)
        for bank in range(utilized_banks_per_channel):
            print("CopyGB K: Bank {}, Row index {}, Op size {}".format(bank*4+ 2, row_index_xk, op_size_xk * 2 ))#check how many banks are accessed
            self.COPY_MIN_GB_only_trace(channel_lst_multi_transformer_block, bank*4+2, row_index_xk, op_size_xk * 2 ) ##OUTPUT Bank=  : bank_group_index=2
            rd_k_burst_l += (op_size_xk * 2)
            
            #print("Inverse for K shuffle: Channel list {}, Op size {}".format(channel_lst_multi_transformer_block, op_size_xk * 2 ))  
            #self.MULINT_only_trace(channel_lst_multi_transformer_block, op_size_xk * 2 )
            print("CopyGB K: Bank {}, Total burst till now: {}".format(bank*4+2, rd_k_burst_l))
        self.rd_k_burst = rd_k_burst_l*channel_lst_multi_transformer_block

        self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)
        self.load_for_rd_k_burst += num_transformer_blocks_per_device* input_vector_EWMUL_utilized_banks* input_vector_EWMUL_length// self.n_repeat * 2 // self.burst_length
        
        print("After LOAD for EWMUL input for K, trace the EWMUL input channel {} ,utilized banks {} , row {},  opsize {}:".format(channels_required, input_vector_EWMUL_utilized_banks, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2))

        print("EQUIVALENCE {}, EQUIVALENCE {}".format(self.wr_q_burst,  wr_q_burst_l))
        print("EQUIVALENCE {}, EQUIVALENCE {}".format(self.wr_k_burst,  wr_k_burst_l))
        print("EQUIVALENCE {}, EQUIVALENCE {}".format(self.rd_q_burst,  rd_q_burst_l))
        print("EQUIVALENCE {}, EQUIVALENCE {}".format(self.rd_k_burst,  rd_k_burst_l))

        print("================================ END OF ROT EMB TRACE ==================================")