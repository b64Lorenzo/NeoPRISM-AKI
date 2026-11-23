"""CLI registration for model-input generation."""
from src.preprocessing.inputsModel.config import load_config
from src.preprocessing.inputsModel.builder import build

def run(args):
    result=build(load_config(args.config),require_labels=not args.unlabeled)
    print(f"Wrote {result['patients']} patient rows to {result['input_file']}")
    return 0

def register_parser(subparsers):
    parser=subparsers.add_parser('inputs-model',help='Create patient-level model input CSV')
    parser.add_argument('--config',default='config/inputsModel.ini')
    parser.add_argument('--unlabeled',action='store_true',help='Create feature rows for novel patients without labels')
    parser.set_defaults(handler=run)
