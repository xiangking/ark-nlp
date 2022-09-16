def get_entity_bio(seq, id2label):
    """
    从BIO标注序列中解码出实体
    
    Args:
        seq (list): BIO标注的序列
        id2label (dict): 标注ID对应的标签映射字典, 例如: {0: 'B-LOC', 1: 'I-LOC'}
        
    Returns:
        list: (entity_type, entity_start_idx, entity_end_idx)三元组数组
        
    Example:
        .. code-block::
        
            seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
            get_entity_bio(seq)
            # output: [['PER', 0, 1], ['LOC', 3, 3]]
    """  # noqa: ignore flake8"

    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]

    return chunks


def get_entity_bios(seq, id2label):
    """
    从BIOS标注序列中解码出实体
    
    Args:
        seq (list): BIOS标注的序列
        id2label (dict): 标注ID对应的标签映射字典, 例如: {0: 'B-LOC', 1: 'I-LOC'}
        
    Returns:
        list: (entity_type, entity_start_idx, entity_end_idx)三元组数组
        
    Example:
        .. code-block::
        
            seq = seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
            get_entity_bios(seq)
            # output: [['PER', 0,1], ['LOC', 3, 3]]
    """  # noqa: ignore flake8"

    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]

    return chunks


def get_entities(seq, id2label=None, markup='bios'):
    """
    从标注序列中解码出实体
    
    Args:
        seq (list): 标注的序列
        id2label (dict): 标注ID对应的标签映射字典, 例如: {0: 'B-LOC', 1: 'I-LOC'}
        markup (dict): 序列标注方式: ['bio', 'bios']       
        
    Returns:
        list: (entity_type, entity_start_idx, entity_end_idx)三元组数组
        
    Example:
        .. code-block::
        
            seq = seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
            get_entities(seq)
            # output: [['PER', 0,1], ['LOC', 3, 3]]
    """  # noqa: ignore flake8"

    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    elif markup == 'bios':
        return get_entity_bios(seq, id2label)
    else:
        raise ValueError("The markup does not exist")

def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text

def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)

def get_entities_for_w2ner(instance, l):
    """
    从标注序列中解码出实体

    Args:
        instance (list): 关系分类矩阵, 形如：max_seq_len * max_seq_len
        l (list): 文本长度

    Returns:
        list: ['i-j-k-#-label', ...], 参考utils/span_metric/convert_index_to_text
    """  # noqa: ignore flake8"

    forward_dict = {}
    head_dict = {}
    ht_type_dict = {}
    for i in range(l):
        for j in range(i + 1, l):
            if instance[i, j] == 1:
                if i not in forward_dict:
                    forward_dict[i] = [j]
                else:
                    forward_dict[i].append(j)
    for i in range(l):
        for j in range(i, l):
            if instance[j, i] > 1:
                ht_type_dict[(i, j)] = instance[j, i]
                if i not in head_dict:
                    head_dict[i] = {j}
                else:
                    head_dict[i].add(j)

    predicts = []

    def find_entity(key, entity, tails):
        entity.append(key)
        if key not in forward_dict:
            if key in tails:
                predicts.append(entity.copy())
            entity.pop()
            return
        else:
            if key in tails:
                predicts.append(entity.copy())
        for k in forward_dict[key]:
            find_entity(k, entity, tails)
        entity.pop()

    for head in head_dict:
        find_entity(head, [], head_dict[head])

    predicts = set(
        [convert_index_to_text(x, ht_type_dict[(x[0], x[-1])]) for x in predicts])

    return predicts

