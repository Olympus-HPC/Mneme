#include "helper.hpp"

std::vector<std::string> read_fasta(std::string in_file, int &largest){
    std::ifstream in_stream(in_file);
    std::string in_line;
    std::vector<std::string> in_sequences;
    int max_size = 0;

    if(in_stream.is_open())
    {
      while(getline(in_stream, in_line))
      {
          if(in_line[0] == '>'){
            continue;
          }else{
            std::string seq = in_line;
            in_sequences.push_back(seq);
            if(max_size < seq.length())
                max_size = seq.length();
          }
      }
      in_stream.close();
    }
    largest = max_size;
    return in_sequences;
}

void read_locassm_data(std::vector<CtgWithReads> *data_in, std::string fname, 
uint32_t& max_ctg_size, uint32_t& total_r_reads, uint32_t& total_l_reads, uint32_t& max_read_size, uint32_t& max_r_count, uint32_t& max_l_count){
    std::ifstream f(fname);
    std::string line;
    max_ctg_size = 0, total_l_reads = 0, total_r_reads = 0, max_read_size = 0, max_read_size = 0, max_l_count = 0, max_r_count = 0;
    while(getline(f, line)) {
      std::stringstream ss(line);
      CtgWithReads temp_in;
      int lsize = 0, rsize = 0;
      //CtgWithReads ctg_with_reads;
      ss >> temp_in.cid >> temp_in.seq >> temp_in.depth >> lsize >> rsize;
      total_l_reads += lsize;
      total_r_reads += rsize;
      if(max_r_count < rsize)
        max_r_count = rsize;
      if(max_l_count < lsize)
        max_l_count = lsize;
      temp_in.max_reads = max(lsize, rsize);
      if (temp_in.seq.size() > max_ctg_size)
          max_ctg_size = temp_in.seq.size();
      for (int i = 0; i < lsize; i++) {
        ReadSeq read_seq;
        ss >> read_seq.read_id >> read_seq.seq >> read_seq.quals;
        if(read_seq.seq.size() > max_read_size)
          max_read_size = read_seq.seq.size();
        temp_in.reads_left.push_back(read_seq);
      }
      for (int i = 0; i < rsize; i++) {
        ReadSeq read_seq;
        ss >> read_seq.read_id >> read_seq.seq >> read_seq.quals;
        if(read_seq.seq.size() > max_read_size)
          max_read_size = read_seq.seq.size();
        temp_in.reads_right.push_back(read_seq);
      }
      //ctgs_map->insert({ctg_with_reads.cid, ctg_with_reads});
      data_in->push_back(temp_in);
    }
    print_vals("inside max:", max_r_count);
  }

  void print_loc_data(std::vector<CtgWithReads> *data_in){
        for(int i = 0; i < (*data_in).size(); i++){
        print_vals("contig_id: ", (*data_in)[i].cid, "\n seq: ", (*data_in)[i].seq, "\n depth: ", (*data_in)[i].depth, "\n right: ", (*data_in)[i].reads_right.size(), "\n left: ",(*data_in)[i].reads_left.size());
        print_vals("**READS**");
        for(int j = 0; j< (*data_in)[i].reads_left.size(); j++){
            ReadSeq read = (*data_in)[i].reads_left[j];
            print_vals(read.read_id, " ", read.seq, " ", read.quals);
            print_vals("READ_SIZE:", read.seq.size(), "QUALS_SIZE:", read.quals.size());
        }
        for(int j = 0; j< (*data_in)[i].reads_right.size(); j++){
            ReadSeq read = (*data_in)[i].reads_right[j];
            print_vals(read.read_id, " ", read.seq, " ", read.quals);
            print_vals("READ_SIZE:", read.seq.size(), "QUALS_SIZE:", read.quals.size());
        }

    }
  }



