class Predictor:
    def __call__(self, x):
        raise NotImplementedError

    def train(self, train):
        raise NotImplementedError


class ModelPredictor(Predictor):
    def __init__(self, model):
        self.model = model
        self.classifier_head = model.classifier_head
        self.period_head = model.period_head
        self.cont_head = model.cont_head

    def __call__(self, x):
        return self.model(x)

    def train(self, train):
        self.model.train(train)


class AverageEnsemblePredictor(Predictor):
    # cannot be used for train
    def __init__(self, models):
        assert len(models) > 0
        self.models = models
        self.classifier_head = models[0].classifier_head
        self.period_head = models[0].period_head
        self.cont_head = models[0].cont_head

    def __call__(self, x):
        sum_period_pred = None
        sum_class_pred = None
        sum_cont_pred = None
        for i, model in enumerate(self.models):
            class_pred, period_pred, cont_pred = model(x)

            if i == 0:
                sum_class_pred = class_pred.detach()
                sum_period_pred = period_pred.detach()
                sum_cont_pred = cont_pred.detach()
            else:
                sum_class_pred += class_pred.detach()
                sum_period_pred += period_pred.detach()
                sum_cont_pred += cont_pred.detach()

        return sum_class_pred / len(self.models), sum_period_pred / len(self.models), sum_cont_pred / len(self.models)

    def train(self, train):
        assert train is False
        for model in self.models:
            model.train(train)


class AverageEnsemblePredictorNoCont(Predictor):
    # cannot be used for train
    def __init__(self, models):
        assert len(models) > 0
        self.models = models
        self.classifier_head = models[0].classifier_head
        self.period_head = models[0].period_head

    def __call__(self, x):
        sum_period_pred = None
        sum_class_pred = None
        for i, model in enumerate(self.models):
            class_pred, period_pred = model(x)

            if i == 0:
                sum_class_pred = class_pred.detach()
                sum_period_pred = period_pred.detach()
            else:
                sum_class_pred += class_pred.detach()
                sum_period_pred += period_pred.detach()

        return sum_class_pred / len(self.models), sum_period_pred / len(self.models)

    def train(self, train):
        assert train is False
        for model in self.models:
            model.train(train)