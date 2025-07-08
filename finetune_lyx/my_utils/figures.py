import matplotlib.pyplot as plt


from my_utils.json_tools import load_json

def all_loss(train_path='finetune_lyx/data/train_data.json', test_path='finetune_lyx/data/test_loss.json'):
    data = load_json(train_path)
    loss = data['train_loss']
    eval_step = data['eval_steps']
    eval_loss = data['eval_loss']
    l = len(loss)

    test_loss = load_json(test_path)
    max_test_loss = max(test_loss)
    avg_test_loss = sum(test_loss) / len(test_loss)

    plt.plot(loss, label='train loss')
    plt.plot(eval_step, eval_loss, label='eval loss')
    plt.plot(range(l), [max_test_loss] * l, linestyle='--', label='max test loss(reference)')
    plt.plot(range(l), [avg_test_loss] * l, linestyle='--', label='avg test loss(reference)')
    plt.legend()
    plt.title('Training History and Model Performance')
    plt.xlabel('Epoch')
    plt.xlim(0, 275)
    plt.ylabel('Loss')
    plt.ylim(-0.5, 4)
    plt.show()

    return max_test_loss, avg_test_loss

def rough_hist(file_path='finetune_lyx/data/rouge_hist_old.json'):
    rough:list[dict] = load_json(file_path)
    l = len(rough)
    rougeL_fmeasure = [d['rougeL_fmeasure'] for d in rough]
    rougeL_precision = [d['rougeL_precision'] for d in rough]
    rougeL_recall = [d['rougeL_recall'] for d in rough]
    for i in range(l, 0, -1):
        idx = i - 1
        if rougeL_fmeasure[idx] == 0 and rougeL_precision[idx] == 0 and rougeL_recall[idx] == 0:
            _ = rougeL_fmeasure.pop(idx)
            _ = rougeL_precision.pop(idx)
            _ = rougeL_recall.pop(idx)
    
    l = len(rougeL_fmeasure)
    x = range(l)
    width = 0.2  # 柱子宽度
    plt.bar([i-width for i in x], rougeL_fmeasure, width, label='f')
    plt.bar(x, rougeL_precision, width, label='precision')
    plt.bar([i+width for i in x], rougeL_recall, width, label='recall')
    plt.xticks(x)
    plt.xlabel('Samples')
    plt.ylabel('Score')
    plt.legend()
    plt.title('ROUGH scores')
    plt.show()