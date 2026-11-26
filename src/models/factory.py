"""Numerical classifier construction without runtime, data or filesystem coupling."""


def model_class(model_type):
    if model_type == "multilayer_perceptron":
        from src.models.neural_network import MultiLayerPerceptron

        return MultiLayerPerceptron
    if model_type == "logistic_regression":
        from src.models.logistic_regression import LogisticRegression

        return LogisticRegression
    if model_type == "multi_output_perceptron":
        from src.models.multi_output_perceptron import MultiOutputPerceptron

        return MultiOutputPerceptron
    raise ValueError("Unsupported numerical model type")


def create_model(model_type, constructor):
    try:
        return model_class(model_type)(**constructor)
    except TypeError as error:
        raise ValueError("Invalid model constructor settings") from error


def fit_numeric(model_type, x, y, constructor, fit_options):
    model = create_model(model_type, constructor)
    model.fit(x, y, **fit_options)
    return model.to_dict()


def restore_model(model_type, payload):
    return model_class(model_type).from_dict(payload)
