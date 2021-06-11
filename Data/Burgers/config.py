import argparse, textwrap
formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=50)

def cla():
    parser = argparse.ArgumentParser(description='list of arguments',formatter_class=formatter)

    parser.add_argument('--Nx', type=int, default=64, help=textwrap.dedent('''Number of spatial nodes  (default: %(default)s). ''')) 
    parser.add_argument('--Nt', type=int, default=64, help=textwrap.dedent('''Number of temporal nodes (default: %(default)s). '''))
    parser.add_argument('--Tfinal', type=float, default=0.2, help=textwrap.dedent('''Final time of simulation (default: %(default)s). '''))
    parser.add_argument('--Nu', type=int, default=10000, help=textwrap.dedent('''Number of realizations of the solution solved for(default: %(default)s). '''))
    parser.add_argument('--Nu_save', type=int, default=1000, help=textwrap.dedent('''Number of realizations of the solution saved to file(default: %(default)s). This must be smaller than Nu. '''))
    parser.add_argument('--ic_type', type=str, required=True, choices=['SineExp','Jump'] , help=textwrap.dedent('''Type of initial condition.'''))
    parser.add_argument('--base_params', type=float, nargs='+', default=[], help=textwrap.dedent('''List of base parameters used to create stochasticity (default: %(default)s).'''))
    parser.add_argument('--visc',type=float,default=0.1, help=textwrap.dedent('''Viscosity coefficient (default: %(default)s). ''')) 
    parser.add_argument('--save_suffix', type=str, default='', help=textwrap.dedent('''Additional suffix to create data directory (default: %(default)s). ''')) 
    parser.add_argument('--seed_no', type=int, default=1008, help=textwrap.dedent('''Random seed (default: %(default)s). '''))

    return parser.parse_args()


