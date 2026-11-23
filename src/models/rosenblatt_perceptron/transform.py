"""Training-only fitted preprocessing with numeric median imputation and categorical one-hot.

Never learn medians/categories from validation or novel patients.
"""
from dataclasses import dataclass
from statistics import median,mean,pstdev
from math import isfinite

@dataclass
class Transformer:
    features: list[str]
    specs: list[dict]

    @classmethod
    def fit(cls,rows,features):
        if not rows: raise ValueError('No training rows')
        specs=[]
        for key in features:
            observed=[r.get(key,'').strip() for r in rows if r.get(key,'').strip()]
            if not observed: raise ValueError(f'All training values missing for {key}')
            numeric=True
            try:
                values=[float(v) for v in observed]
                numeric=all(isfinite(v) for v in values)
            except ValueError: numeric=False
            if numeric:
                m=median(values); avg=mean(values);sd=pstdev(values) or 1.0
                specs.append({'type':'numeric','median':m,'mean':avg,'scale':sd})
            else:
                cats=sorted(set(observed))
                specs.append({'type':'categorical','categories':cats})
        return cls(list(features),specs)

    def transform_one(self,row):
        result=[]
        for key,spec in zip(self.features,self.specs):
            raw=(row.get(key) or '').strip()
            if spec['type']=='numeric':
                try: val=float(raw) if raw else spec['median']
                except ValueError: raise ValueError(f'Non-numeric value for {key}')
                if not isfinite(val): raise ValueError(f'Non-finite value for {key}')
                result.extend([(val-spec['mean'])/spec['scale'],float(not raw)])
            else:
                result.extend([float(raw==cat) for cat in spec['categories']])
                result.append(float(not raw))
        return result

    def transform(self,rows): return [self.transform_one(r) for r in rows]

    def to_dict(self): return {'features':self.features,'specs':self.specs}

    @classmethod
    def from_dict(cls,data):
        if not isinstance(data,dict) or not isinstance(data.get('features'),list) or not isinstance(data.get('specs'),list):
            raise ValueError('Invalid transformer schema')
        if len(data['features'])!=len(data['specs']): raise ValueError('Transformer schema mismatch')
        return cls(data['features'],data['specs'])
