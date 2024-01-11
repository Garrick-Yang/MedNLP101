







# 举例，将一样的变量区分出来，后边加上“_IE”“_CLASSIFIER”"_CDN”等等
MODEL_CLASS_CLASSIFIER = {
    'bert': (BertTokenizer, BertForSequenceClassification),
    'roberta': (BertTokenizer, BertForSequenceClassification),
    'albert': (BertTokenizer, AlbertForSequenceClassification),
    'zen': (BertTokenizer, ZenForSequenceClassification)
}


MODEL_CLASS_IE = {
    'bert': (BertTokenizer, BertModel),
    'roberta': (BertTokenizer, BertModel),
    'albert': (BertTokenizer, AlbertModel),
    'zen': (BertTokenizer, ZenModel)
}