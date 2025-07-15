#include "util.h"

program_options parse_arguments(int argc, char *argv[])
{
  program_options op;
  int c;

  static struct option long_options[] =
  {
    {"device",required_argument,0,'d'},
    {"help",no_argument,0,'h'},
    {"infile",required_argument,0,'i'},
    {"approx",required_argument,0,'k'},
    {"printscores",optional_argument,0,'p'},
    {"verify",no_argument,0,'v'},
    {0,0,0,0} //Terminate with null
  };

  int option_index = 0;

  while((c = getopt_long(argc,argv,"hi:k:p::v",long_options,&option_index)) != -1)
  {
    switch(c)
    {
      case 'h':
        std::cout << "Usage: " << argv[0] << " -i <input graph file> [-v verify GPU calculation] [-p <output file> print BC scores]" << std::endl;  
        exit(0);

      case 'i':
        op.infile = optarg;
        break;

      case 'k':
        op.approx = true;
        op.k = atoi(optarg);
        break;

      case 'p':
        op.printBCscores = true;
        op.scorefile = optarg;
        break;

      case 'v':
        op.verify = true;
        break;

      case '?': //Invalid argument: getopt will print the error msg itself

        exit(-1);

      default: //Fatal error
        std::cerr << "Fatal error parsing command line arguments. Terminating." << std::endl;
        exit(-1);
    }
  }

  if(op.infile == NULL)
  {
    std::cerr << "Command line error: Input graph file is required. Use the -i switch." << std::endl;
  }

  return op;
}

void verify(graph g, const std::vector<float> bc_cpu, const std::vector<float> bc_gpu)
{
  double error = 0;
  double max_error = 0;
  for(int i=0; i<g.n; i++)
  {
    double current_error = fabs(bc_cpu[i] - bc_gpu[i]);
    error += current_error*current_error;
    if(current_error > max_error)
    {
      max_error = current_error;
    }
  }
  error = error/(float)g.n;
  error = sqrt(error);
  std::cout << "RMS Error: " << error << std::endl;
  std::cout << "Maximum error: " << max_error << std::endl;
}
