from transformer import *
from datetime import datetime
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import os
import math

# Load text data and encode it all.
# TODO: Use HuggingFace Dataset and Torch DataLoader
with open("chat_simplified.txt", 'r', encoding='utf-8') as f:
    text = f.read()
data = encode(text).to(device)
del text

# Separate data in train and test split
n = int(0.9*len(data))
dataset_split_seed = 0
train_data = data[:n]
val_data = data[n:]

# Random chunks in batches
def get_batch(tipo='train'):
    data = train_data if tipo == 'train' else val_data
    
    chunks = list(data.split((block_size+1) * batch_size))
    
    random.seed(dataset_split_seed)
    random.shuffle(chunks)
    for chunk in chunks:        
        ix = torch.arange(0, chunk.shape[-1], step=block_size+1, dtype=int)
    
        x = [chunk[i:i+block_size+1] for i in ix]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).to(device)
                
        yield x

data_train = get_batch("train")
data_test = get_batch("test")

def get_train_size(tipo="train"):
    data = train_data if tipo == 'train' else val_data
    return math.ceil(data.shape[-1] / ((block_size+1) * batch_size))

def plot_history(model, train_history, loss_type):
    fig, ax = plt.subplots(figsize=(8,5))

    ax.set_ylim([2.0, 4.0])

    # Plot para el entrenamiento y la pérdida de prueba
    ax.plot(train_history['iteration'], train_history['train_loss'], label=f'Train loss', marker='o')
    ax.plot(train_history['iteration'], train_history['test_loss'], label=f'Test loss', marker='o')

    # Líneas verticales para cada época
    for x in train_history.groupby("epoch").max()["iteration"][:-1]:
        ax.axvline(x=x)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss per Iteration and Epoch')
    ax.legend()
    ax.grid(True)
    
    sep=',\n'
    text = f"""
{str(type(model)).split(".")[-1].split("'")[0]}

vocab_size = {model.vocab_size}
n_embed = {model.n_embed}
n_layers = {model.n_layers},
{sep.join([f'{k} = {v}' for k,v in model.layer_props.items()])},
parameters = {model.numero_parametros()}

block_size = {block_size}
batch_size = {batch_size}
dropout = {model.dropout}
loss = '{loss_type}'
data_seed = {dataset_split_seed}
    """    
    ax.text(1.02, 1.0, text, transform=ax.transAxes, fontsize=12, verticalalignment="top", horizontalalignment="left")
    
    fig.tight_layout()
    return fig

class TrainingRun:
    def __init__(self, model, **props):
        self.model = model
        
        self.constant_props = {}
        self.varing_props = []
        
        for prop_name, prop_value in props.items():
            if isinstance(prop_value, list):
                self.varing_props.append((prop_name, prop_value))
            else:
                self.constant_props[prop_name] = prop_value
    
    def train_iteration(self, m, epochs, loss_type="mean", initial_lr=1e-2, final_lr=None):
        if final_lr is None:
            final_lr = initial_lr / 10.0
        gamma = math.pow(final_lr/initial_lr, 1.0/epochs)
    
        # Length of training run
        epochs = range(epochs)
        # TEST
        eval_interval = get_train_size() // 4
        # TEST
        #eval_interval = len(data_train) // 4

        # Initialize training history
        columns = ['epoch', 'iteration', 'train_loss', 'test_loss', 'learning_rate']
        train_history = pd.DataFrame(columns=columns)

        # Set optimizer and scheduler
        optimizer = torch.optim.AdamW(m.parameters(), lr=initial_lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, verbose=True)
        # Estaba así antes
        #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8, verbose=True)
        
        # Start traning model for all epochs
        for epoch in epochs:
            print(f"Epoch {epoch}")
            # TEST
            data = get_batch("train")
            #data = iter(data_train)
            # TEST
            epoch_history = []

            for i, batch in enumerate(data):
                if i % eval_interval == 0 and (i != 0 or epoch == 0):
                    losses = m.estimate_loss({
                        "train": get_batch("train"), 
                        "test": get_batch("test")
                    }, loss_type=loss_type, test_iters=2)
                    # TEST
                    print(f"Step {i}/{get_train_size()} train loss ({losses['train']}), test loss ({losses['test']})")
                    #print(f"Step {i}/{len(data_train)} train loss ({losses['train']}), test loss ({losses['test']})")
                    # TEST
                    
                    new_row = {
                        'epoch': epoch, 
                        'iteration': i + get_train_size()*epoch,
                        'train_loss': losses['train'],
                        'test_loss': losses['test'],
                        'learning_rate': lr_scheduler.get_last_lr()[0]
                    }
                    train_history = pd.concat([train_history, pd.DataFrame.from_records([new_row]).astype(train_history.dtypes)])
                    del new_row, losses
                                
                x,y = batch[:,:-1], batch[:,:]
                logits, loss = m(x, y)
                del logits, x, y
                optimizer.zero_grad(set_to_none=True)
                loss[loss_type].backward()
                del loss
                optimizer.step()
            lr_scheduler.step()
        return train_history
        
    def train_all(self, folder, epochs, loss_type="mean", initial_lr=1e-2, final_lr=None):
        print(f"Training constant values: {self.constant_props}")
        
        permute = [x[1] for x in self.varing_props]
        for values in itertools.product(*permute):
            iteration_props = {self.varing_props[i][0]:x for i,x in enumerate(values)}
            m = self.model(**self.constant_props, **iteration_props).to(device)
            parameters = m.numero_parametros()
            print(f"Training iteration with {iteration_props}; {parameters} parameters")
            
            train_history = self.train_iteration(m, epochs, loss_type, initial_lr, final_lr)
                        
            iteration_name = "-".join([f"{name}_{prop}" for name,prop in iteration_props.items()])
            print(f"Saving as '{iteration_name}'")
            
            folder_name = os.path.join(folder, iteration_name)
            os.makedirs(folder_name, exist_ok=True)

            checkpoint_filename = os.path.join(folder_name, f'checkpoint.pth')
            m.guardar_parametros(checkpoint_filename)
            
            csv_filename = os.path.join(folder_name, 'history.csv')
            train_history.to_csv(csv_filename, index=False)

            plot_filename = os.path.join(folder_name, 'plot.png')
            plot_history(m, train_history, loss_type).savefig(plot_filename)

            config_filename = os.path.join(folder_name, f'config.txt')
            with open(config_filename, 'w') as f:
                f.write(str(m))
                f.write(f"\n{m.numero_parametros()} parámetros")

training_run = TrainingRun(
    Transformer,
    vocab_size=len(sp),
    window_size=(16,8,4,2),
    window_stride=(1,2,4,8),
    #window_size=  ((16,8),(16,8),(16,8),(16,8)),
    #window_stride=((1, 1),(1, 4),(1, 4),(1, 4)),
    encoding="relative",
    n_embed=64,
    n_layers=4,
    n_head=4,
    dropout=0.4
)
training_run.train_all(folder="checkpoints/small_test", epochs=5)
