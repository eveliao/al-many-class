from transformers import DataProcessor, InputExample
import pickle
import os

def active_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return {"acc": simple_accuracy(preds, labels)}

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

class TnewsProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.label_list = None
        self.label_map = None
        self.data_dir = '../data'
        self._load_data()

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self):
        """See base class."""
        return self.train_examples

    def get_dev_examples(self):
        """See base class."""
        return self.dev_examples

    def get_test_examples(self):
        """See base class."""
        return self.test_examples

    def get_labels(self):
        """See base class."""
        return self.label_list

    def get_label_map(self):
        return self.label_map

    def get_label_map_inverse(self):
        return self.label_map_inverse

    # def get_encoder(self):
    #     return self.label_encoder

    def _create_examples(self, x, y, set_type):
        examples = []
        if x is None:
            return examples
        for (i, text) in enumerate(x):
            guid = "%s-%s" % (set_type, i)
            text_a = text
            label = y[i]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _load_data(self):

        with open(
                os.path.join(self.data_dir,
                             'tnews/tnews.{}.pkl'.format(self.num_classes)), 'rb') as f:
            dic = pickle.load(f)

        train_x = dic['train_x']
        train_y = dic['train_y']
        test_x = dic['test_x']
        test_y = dic['test_y']

        dev_x, dev_y = None, None
        if 'dev_x' in dic:
            dev_x = dic['dev_x']
            dev_y = dic['dev_y']

        self.train_examples = self._create_examples(train_x, train_y, 'train')
        self.test_examples = self._create_examples(test_x, test_y, 'test')
        self.dev_examples = self._create_examples(dev_x, dev_y, 'dev')

        label_list = set()
        for label in train_y:
            label_list.add(label)
        for label in test_y:
            label_list.add(label)
        if dev_y is not None:
            for label in dev_y:
                label_list.add(label)

        label_list = sorted(list(label_list))

        self.label_list = label_list
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.label_map_inverse = {i: label for i, label in enumerate(label_list)}


class yanjingProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.label_list = None
        self.label_map = None
        self.data_dir = '../data'
        self._load_data()

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self):
        """See base class."""
        return self.train_examples

    def get_dev_examples(self):
        """See base class."""
        return self.dev_examples

    def get_test_examples(self):
        """See base class."""
        return self.test_examples

    def get_labels(self):
        """See base class."""
        return self.label_list

    def get_label_map(self):
        return self.label_map

    def get_label_map_inverse(self):
        return self.label_map_inverse

    # def get_encoder(self):
    #     return self.label_encoder

    def _create_examples(self, x, y, set_type):
        examples = []
        if x is None:
            return examples
        for (i, text) in enumerate(x):
            guid = "%s-%s" % (set_type, i)
            text_a = text
            label = y[i]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _load_data(self):

        with open(
                os.path.join(self.data_dir,
                             'yanjing/yanjing.{}.pkl'.format(self.num_classes)), 'rb') as f:
            dic = pickle.load(f)

        train_x = dic['train_x']
        train_y = dic['train_y']
        test_x = dic['test_x']
        test_y = dic['test_y']

        dev_x, dev_y = None, None
        if 'dev_x' in dic:
            dev_x = dic['dev_x']
            dev_y = dic['dev_y']

        self.train_examples = self._create_examples(train_x, train_y, 'train')
        self.test_examples = self._create_examples(test_x, test_y, 'test')
        self.dev_examples = self._create_examples(dev_x, dev_y, 'dev')

        label_list = set()
        for label in train_y:
            label_list.add(label)
        for label in test_y:
            label_list.add(label)
        if dev_y is not None:
            for label in dev_y:
                label_list.add(label)

        label_list = sorted(list(label_list))
        self.label_list = label_list
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.label_map_inverse = {i: label for i, label in enumerate(label_list)}


class ReutersProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.label_list = None
        self.label_map = None
        self.data_dir = '../data'
        self._load_data()

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self):
        """See base class."""
        return self.train_examples

    def get_dev_examples(self):
        """See base class."""
        return self.dev_examples

    def get_test_examples(self):
        """See base class."""
        return self.test_examples

    def get_labels(self):
        """See base class."""
        return self.label_list

    def get_label_map(self):
        return self.label_map

    def get_label_map_inverse(self):
        return self.label_map_inverse

    # def get_encoder(self):
    #     return self.label_encoder

    def _create_examples(self, x, y, set_type):
        examples = []
        if x is None:
            return examples
        for (i, text) in enumerate(x):
            guid = "%s-%s" % (set_type, i)
            text_a = text
            label = y[i]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _load_data(self):

        with open(
                os.path.join(self.data_dir,
                             'reuters/reuters.{}.pkl'.format(self.num_classes)), 'rb') as f:
            dic = pickle.load(f)

        x = dic['x']
        y = dic['y']
        test_size = int(len(y) * 0.2)
        test_x, test_y = x[:test_size], y[:test_size]
        train_x, train_y = x[test_size:], y[test_size:]

        dev_x, dev_y = None, None
        if 'dev_x' in dic:
            dev_x = dic['dev_x']
            dev_y = dic['dev_y']

        self.train_examples = self._create_examples(train_x, train_y, 'train')
        self.test_examples = self._create_examples(test_x, test_y, 'test')
        self.dev_examples = self._create_examples(dev_x, dev_y, 'dev')

        label_list = set()
        for label in train_y:
            label_list.add(label)
        for label in test_y:
            label_list.add(label)
        if dev_y is not None:
            for label in dev_y:
                label_list.add(label)

        label_list = sorted(list(label_list))
        self.label_list = label_list
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.label_map_inverse = {i: label for i, label in enumerate(label_list)}

class SearchProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.label_list = None
        self.label_map = None
        self.data_dir = '../data'
        self._load_data()

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self):
        """See base class."""
        return self.train_examples

    def get_dev_examples(self):
        """See base class."""
        return self.dev_examples

    def get_test_examples(self):
        """See base class."""
        return self.test_examples

    def get_labels(self):
        """See base class."""
        return self.label_list

    def get_label_map(self):
        return self.label_map

    def get_label_map_inverse(self):
        return self.label_map_inverse

    # def get_encoder(self):
    #     return self.label_encoder

    def _create_examples(self, x, y, set_type):
        examples = []
        if x is None:
            return examples
        for (i, text) in enumerate(x):
            guid = "%s-%s" % (set_type, i)
            text_a = text
            label = y[i]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _load_data(self):

        with open(
                os.path.join(self.data_dir,
                             'SearchSnippets/SearchSnippets.{}.pkl'.format(self.num_classes)), 'rb') as f:
            dic = pickle.load(f)

        x = dic['x']
        y = dic['y']
        test_size = int(len(y) * 0.2)
        test_x, test_y = x[:test_size], y[:test_size]
        train_x, train_y = x[test_size:], y[test_size:]

        dev_x, dev_y = None, None
        if 'dev_x' in dic:
            dev_x = dic['dev_x']
            dev_y = dic['dev_y']

        self.train_examples = self._create_examples(train_x, train_y, 'train')
        self.test_examples = self._create_examples(test_x, test_y, 'test')
        self.dev_examples = self._create_examples(dev_x, dev_y, 'dev')

        label_list = set()
        for label in train_y:
            label_list.add(label)
        for label in test_y:
            label_list.add(label)
        if dev_y is not None:
            for label in dev_y:
                label_list.add(label)

        label_list = sorted(list(label_list))
        self.label_list = label_list
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.label_map_inverse = {i: label for i, label in enumerate(label_list)}


class BookProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.label_list = None
        self.label_map = None
        self.data_dir = '../data'
        self._load_data()

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self):
        """See base class."""
        return self.train_examples

    def get_dev_examples(self):
        """See base class."""
        return self.dev_examples

    def get_test_examples(self):
        """See base class."""
        return self.test_examples

    def get_labels(self):
        """See base class."""
        return self.label_list

    def get_label_map(self):
        return self.label_map

    def get_label_map_inverse(self):
        return self.label_map_inverse

    # def get_encoder(self):
    #     return self.label_encoder

    def _create_examples(self, x, y, set_type):
        examples = []
        if x is None:
            return examples
        for (i, text) in enumerate(x):
            guid = "%s-%s" % (set_type, i)
            text_a = text
            label = y[i]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _load_data(self):

        with open(
                os.path.join(self.data_dir,
                             'book/book.{}.pkl'.format(self.num_classes)), 'rb') as f:
            dic = pickle.load(f)

        x = dic['x']
        y = dic['y']
        test_size = int(len(y) * 0.2)
        test_x, test_y = x[:test_size], y[:test_size]
        train_x, train_y = x[test_size:], y[test_size:]

        dev_x, dev_y = None, None
        if 'dev_x' in dic:
            dev_x = dic['dev_x']
            dev_y = dic['dev_y']

        self.train_examples = self._create_examples(train_x, train_y, 'train')
        self.test_examples = self._create_examples(test_x, test_y, 'test')
        self.dev_examples = self._create_examples(dev_x, dev_y, 'dev')

        label_list = set()
        for label in train_y:
            label_list.add(label)
        for label in test_y:
            label_list.add(label)
        if dev_y is not None:
            for label in dev_y:
                label_list.add(label)

        label_list = sorted(list(label_list))
        self.label_list = label_list
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.label_map_inverse = {i: label for i, label in enumerate(label_list)}


class NewsProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.label_list = None
        self.label_map = None
        self.data_dir = '../data'
        self._load_data()

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self):
        """See base class."""
        return self.train_examples

    def get_dev_examples(self):
        """See base class."""
        return self.dev_examples

    def get_test_examples(self):
        """See base class."""
        return self.test_examples

    def get_labels(self):
        """See base class."""
        return self.label_list

    def get_label_map(self):
        return self.label_map

    def get_label_map_inverse(self):
        return self.label_map_inverse

    # def get_encoder(self):
    #     return self.label_encoder

    def _create_examples(self, x, y, set_type):
        examples = []
        if x is None:
            return examples
        for (i, text) in enumerate(x):
            guid = "%s-%s" % (set_type, i)
            text_a = text
            label = y[i]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _load_data(self):

        with open(
                os.path.join(self.data_dir,
                             'news/news.{}.pkl'.format(self.num_classes)), 'rb') as f:
            dic = pickle.load(f)

        train_x = dic['train_x']
        train_y = dic['train_y']
        test_x = dic['test_x']
        test_y = dic['test_y']
        dev_x, dev_y = None, None
        if 'dev_x' in dic:
            dev_x = dic['dev_x']
            dev_y = dic['dev_y']

        self.train_examples = self._create_examples(train_x, train_y, 'train')
        self.test_examples = self._create_examples(test_x, test_y, 'test')
        self.dev_examples = self._create_examples(dev_x, dev_y, 'dev')

        label_list = set()
        for label in train_y:
            label_list.add(label)
        for label in test_y:
            label_list.add(label)
        if dev_y is not None:
            for label in dev_y:
                label_list.add(label)

        label_list = sorted(list(label_list))
        self.label_list = label_list
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.label_map_inverse = {i: label for i, label in enumerate(label_list)}


active_processors = {
    'reuters': ReutersProcessor,
    'news': NewsProcessor,
    'Search': SearchProcessor,
    'tnews': TnewsProcessor,
    'yanjing': yanjingProcessor,
    'book': BookProcessor,
}

active_output_modes = {
    'reuters': 'classification',
    'news': 'classification',
    'Search': 'classification',
    'tnews': "classification",
    'yanjing': "classification",
    'book': 'classification',
}

active_lang_dic = {
    'reuters': '../bert_pretrained/uncased_L-12_H-768_A-12',
    'news': '../bert_pretrained/uncased_L-12_H-768_A-12',
    'Search': '../bert_pretrained/uncased_L-12_H-768_A-12',
    'tnews': '../bert_pretrained/chinese_L-12_H-768_A-12',
    'yanjing': '../bert_pretrained/chinese_L-12_H-768_A-12',
    'book': '../bert_pretrained/chinese_L-12_H-768_A-12',
}