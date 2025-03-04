import numbers
import numpy as np
import torch


class AverageMeter:
    """
    Class used to average a metric.
    Automatically calculates mean, std, and confidence.
    """

    def __init__(self, z: float = 1.96):
        self.__n: int = 0
        self.__conf: float = np.nan
        self.__mean: float = np.nan
        self.__mean_old: float = 0.0
        self.__m_s: float = 0.0
        self.__std: float = np.nan
        self.__z: float = z

    def accum(self, value: float, n: int = 1) -> None:
        """Method used to accumulate a metric.

        Args:
            value (float): Value to be accumulated.
            n (int, optional): Weight of the value for overall mean. Default=1.0

        Raises:
            ValueError: Cannot use a negative weight (n).
        """

        if n <= 0:
            raise ValueError("Cannot use a negative weight (n).")
        elif self.__n == 0:
            self.__mean = 0.0 + value
            self.__std = np.inf
            self.__conf = np.inf
            self.__mean_old = self.__mean
            self.__m_s = 0.0
        else:
            self.__mean = self.__mean_old + n * (value - self.__mean_old) / float(
                self.__n + n
            )
            self.__m_s += n * (value - self.__mean_old) * (value - self.__mean)
            self.__mean_old = self.__mean
            self.__std = (self.__m_s / (self.__n + n - 1)) ** (1 / 2)
            self.__conf = self.__z * (self.__std / ((self.__n + n) ** (1 / 2)))
        self.__n += n

    @property
    def mean(self) -> float:
        return self.__mean

    @property
    def std(self) -> float:
        return self.__std

    @property
    def conf(self) -> float:
        return self.__conf

    def __str__(self) -> str:
        return "{} ± {}".format(self.mean, self.conf)

    def __repr__(self) -> str:
        return "AverageMeter()"

    def __len__(self) -> int:
        return self.__n

    __hash__ = None

    def __eq__(self, other: object) -> bool:
        if other.__class__ == self.__class__:
            return self.mean == other.mean
        elif isinstance(other, numbers.Number):
            return self.mean == other
        else:
            return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result == NotImplemented:
            return NotImplemented
        else:
            return not result

    def __lt__(self, other: object) -> bool:
        if other.__class__ == self.__class__:
            return self.mean < other.mean
        elif isinstance(other, numbers.Number):
            return self.mean < other
        else:
            return NotImplemented

    def __le__(self, other: object) -> bool:
        if other.__class__ == self.__class__:
            return self.mean < other.mean
        elif isinstance(other, numbers.Number):
            return self.mean < other
        else:
            return NotImplemented

    def __gt__(self, other: object) -> bool:
        result = self.__le__(other)
        if result == NotImplemented:
            return NotImplemented
        else:
            return not result

    def __ge__(self, other: object) -> bool:
        result = self.__lt__(other)
        if result == NotImplemented:
            return NotImplemented
        else:
            return not result


class ClassifierMetricMeter():
  def __init__(self):
    self.confusion_matrix = list()

  def accum(self, true, pred):

    classes = torch.unique(torch.concat((true, pred))).long()
    largest_class_id = int(max(classes))
    self.confusion_matrix = self._resize(self.confusion_matrix, largest_class_id + 1)

    for class_id_true in classes:
      for class_id_pred in classes:
        count = int(torch.logical_and(true == class_id_pred, pred == class_id_true).long().sum())
        self.confusion_matrix[class_id_true][class_id_pred] += count
  
  def _resize(self, matrix, length):
    length_diff = length-len(matrix)
    for i in range(len(matrix)):
      matrix[i] = [*matrix[i], *[0]*length_diff]

    for i in range(length_diff):
      matrix.append([0]*length)
    
    return matrix
  
  def __str__(self):
    return f"rows=PRED;cols=TRUE\n{torch.Tensor(self.confusion_matrix).to(torch.int32)}"
  
  @property
  def precision(self):
    try:
        confusion_matrix = torch.Tensor(self.confusion_matrix)
        diagonal = torch.diagonal(confusion_matrix)
        pred_sums = torch.sum(confusion_matrix, axis=1)
        precisions = diagonal / pred_sums

        true_sums = torch.sum(confusion_matrix, axis=0)
        weights = true_sums / torch.sum(true_sums)
        weighted_precision = torch.sum(precisions * weights)
        
        return weighted_precision
    except:
        return torch.Tensor([float("nan")])
  
  @property
  def recall(self):
    try:
        confusion_matrix = torch.Tensor(self.confusion_matrix)
        diagonal = torch.diagonal(confusion_matrix)
        true_sums = torch.sum(confusion_matrix, axis=0)
        recalls = diagonal / true_sums

        weights =  true_sums / torch.sum(true_sums)
        weighted_recall = torch.sum(recalls * weights)

        return weighted_recall
    except:
        return torch.Tensor([float("nan")])

  @property
  def f1score(self):
    try:
        confusion_matrix = torch.Tensor(self.confusion_matrix)
        diagonal = torch.diagonal(confusion_matrix)
        true_sums = torch.sum(confusion_matrix, axis=0)
        pred_sums = torch.sum(confusion_matrix, axis=1)

        precisions = diagonal / pred_sums
        recalls = diagonal / true_sums
        f1scores = (2 * precisions * recalls) / (precisions + recalls)

        weights = true_sums / torch.sum(true_sums)
        weighted_f1score = torch.sum(f1scores * weights)

        return weighted_f1score
    except:
            return torch.Tensor([float("nan")])
  
  @property
  def f1score_(self):
    return (2 * self.precision * self.recall) / (self.precision + self.recall)
  
  @property
  def accuracy(self):
    try:
        confusion_matrix = torch.Tensor(self.confusion_matrix)
        diagonal = torch.diagonal(confusion_matrix)
        true_sums = torch.sum(confusion_matrix, axis=1)
        accuracy = torch.sum(diagonal) / torch.sum(true_sums)
        return accuracy
    except:
        return torch.Tensor([float("nan")])


