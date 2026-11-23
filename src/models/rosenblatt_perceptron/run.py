"""CLI train/evaluate and novel-data execution."""
import json
from src.models.rosenblatt_perceptron.training import fit
from src.models.rosenblatt_perceptron.execution.predict import predict_csv

def train_command(args):
    result=fit(args.input,args.output,train_fraction=args.train_fraction,seed=args.seed,
               epochs=args.max_epochs,learning_rate=args.learning_rate,threshold=args.threshold)
    print(json.dumps(result,indent=2));return 0

def predict_command(args):
    count=predict_csv(args.model,args.input,args.output)
    print(f'Wrote {count} predictions to {args.output}');return 0

def register_parser(subparsers):
    t=subparsers.add_parser('rosenblatt-train',help='Train binary AKI perceptron and evaluate holdout')
    t.add_argument('--input',default='results/inputsModel/patients_labeled.csv')
    t.add_argument('--output',default='results/models/rosenblatt_perceptron')
    t.add_argument('--train-fraction',type=float,default=.8)
    t.add_argument('--seed',type=int,default=2026)
    t.add_argument('--max-epochs',type=int,default=100)
    t.add_argument('--learning-rate',type=float,default=None)
    t.add_argument('--threshold',type=float,default=0.0)
    t.set_defaults(handler=train_command)
    x=subparsers.add_parser('rosenblatt-predict',help='Predict novel patients using frozen model')
    x.add_argument('--model',default='results/models/rosenblatt_perceptron/model.json')
    x.add_argument('--input',required=True)
    x.add_argument('--output',default='results/models/rosenblatt_perceptron/novel_predictions.csv')
    x.set_defaults(handler=predict_command)
