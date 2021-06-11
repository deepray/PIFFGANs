import argparse, textwrap
formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=50)


def cla():
    parser = argparse.ArgumentParser(description='list of arguments',formatter_class=formatter)

    # Data parameters
    parser.add_argument('--data_dir', type=str  , required=True, help=textwrap.dedent('''Dataset directory path'''))
    parser.add_argument('--Nu'      , type=int  , default=1000 , help=textwrap.dedent('''Number of realizations of the solution to train with. Must be less than total available samples (default: %(default)s). ''')) 
    parser.add_argument('--Nx1_frac', type=float, default=0.5  , help=textwrap.dedent('''Fraction of x1 spatial observation points observed for u, always including end points (default: %(default)s).'''))
    parser.add_argument('--Nx2_frac', type=float, default=0.5  , help=textwrap.dedent('''Fraction of x2 spatial observation points observed for u, always including end points (default: %(default)s).'''))

    # Network parameters
    parser.add_argument('--g_type'    , type=str,   default='Standard', choices=['Standard','FF'], help=textwrap.dedent('''Type of generator architecture. Standard takes a normal input, while FF learns on Fourier Features (default: %(default)s).'''))
    parser.add_argument('--g_width'   , type=int,   default=50, help=textwrap.dedent('''Width of hidden layers of generator (default: %(default)s).'''))
    parser.add_argument('--g_depth'   , type=int,   default=6, help=textwrap.dedent('''Depth of hidden layers of generator (default: %(default)s).'''))
    parser.add_argument('--d_type'    , type=str,   default='MLP', choices=['MLP','CNN'], help=textwrap.dedent('''Type of discriminator architecture (default: %(default)s).'''))
    parser.add_argument('--d_width'   , type=int,   default=50, help=textwrap.dedent('''Width of hidden layers of discriminator, if using MLP (default: %(default)s).'''))
    parser.add_argument('--d_depth'   , type=int,   default=6, help=textwrap.dedent('''Depth of hidden layers of discriminator, if using MLP (default: %(default)s).'''))
    parser.add_argument('--gp_coef'   , type=float, default=10.0, help=textwrap.dedent('''Gradient penalty coefficient (default: %(default)s).'''))
    parser.add_argument('--n_critic'  , type=int,   default=4, help=textwrap.dedent('''Number of critic updates per generator update (default: %(default)s).'''))
    parser.add_argument('--n_epoch'   , type=int,   default=1000, help=textwrap.dedent('''Maximum number of epochs (default: %(default)s).'''))
    parser.add_argument('--z_dim'     , type=int,   default=1, help=textwrap.dedent('''Latent space dimension (default: %(default)s).'''))
    parser.add_argument('--batch_size', type=int,   default=50, help=textwrap.dedent('''Training batch size (default: %(default)s).'''))
    parser.add_argument('--FF_len'    , type=int,   default=10, help=textwrap.dedent('''Length of Fourier Feature vector, when using d_type=FF (default: %(default)s).'''))
    parser.add_argument('--FF_sigma'  , type=float, default=1.0, help=textwrap.dedent('''Std of Gaussian noise used to generate B, when using d_type=FF (default: %(default)s).'''))

    # PDE parameters
    parser.add_argument('--pde_type'   , type=str,  required=True, choices=['None','LinAdv','Burgers'], help=textwrap.dedent('''Type of PDE model being considered.'''))
    parser.add_argument('--pde_params' , nargs='+', type=float, default=[],help=textwrap.dedent('''List of parameters used to model the PDE equations, such as velocity, viscousity coefficient, etc (default: %(default)s).'''))
    parser.add_argument('--pdecon_coef', nargs='+', type=float, default=[], help=textwrap.dedent('''List of constraint parameters for the PDE residuals (default: %(default)s).'''))
    
    # Output parameters
    parser.add_argument('--savefig_freq', type=int, default=200, help=textwrap.dedent('''Number of epochs after which solution snapshots and generator checkpoints are saves (default: %(default)s).'''))
    parser.add_argument('--save_dir', type=str, default='Results', help=textwrap.dedent('''Name of directory where the results are saved (default: %(default)s).'''))
    parser.add_argument('--seed_no', type=int, default=1008, help=textwrap.dedent('''Random seed (default: %(default)s). '''))

    param_check(parser.parse_args())

    return parser.parse_args()

def param_check(PARAMS):
    
    if PARAMS.pde_type == 'None':
        assert len(PARAMS.pde_params) == 0 , 'Cannot specify pde_params with pde_type=None'  
        assert len(PARAMS.pdecon_coef) == 0, 'Cannot specify pdecon_coef with pde_type=None'  

    if PARAMS.g_type == 'Standard':
        print(f'--- Ignorning FF_len and FF_sigma since g_type == Standard')
    else:
        assert FF_len > 0 , 'FF_len must be a postive integer'    
        assert FF_sigma > 0.0 , 'FF_sigma must be a postive real number'       


