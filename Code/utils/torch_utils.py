import torch as t
from torch import nn

def generate_new_word_from_torch_model(model, start_character, end_character, padding, ctoi, itoc, method = 'greedy'):
    with t.no_grad():
        model.eval()
        name = [start_character] * padding

        while True:
            input = [ctoi[name[len(name) - i-1]] for i in range(padding)]
            input.reverse()
            input = t.Tensor(input).view(1, -1).type(t.LongTensor)
            logits = model(input)
            dist = t.distributions.Categorical(logits=logits)
            
            character = itoc[dist.sample().item()]
            if character == end_character:
                break                

            name.append(character)

        return ''.join(name[padding:])

def calculate_loss_on_batch(model, batch):
    x = batch[:, 0:-1]
    y = batch[:, -1]

    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)

    return loss

def calculate_loss_on_data_loader(model, loader):
    total_loss = 0
    n = 0
    for batch in loader:
        loss = calculate_loss_on_batch(model, batch)
        total_loss += loss*len(batch)
        n += len(batch)

    return total_loss / n