import torch
import torch.nn as nn
from tqdm import tqdm # to  show the progress bars

# the loss function
def loss_fn(outputs, targets):
    '''
    gets the outputs and actual labels
    and returns the loss
    '''
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1,1))

def train_fn(datal_loader, model,
        optimizer, device, schedular):

    # train the model
    model.train()

    for bi, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        # start parsing from dataloader
        ids = data['ids']
        token_type_ids = data['token_type_ids']
        mask = data['mask']
        targets =  data['targets']

        # set all the  parameters to device
        # lets just skip this section for now
        # as we are already operating on CPU device

        # set optimzier to zero_grad() initially
        optimizer.zero_grad()

        # get model outputs
        outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
                )

        # calculate the loss
        loss = loss_fn(outputs, targets)
        # apply backward pass
        loss.backward()
        # apply optimizer step-wise
        optimzer.step()
        #  apply schedular step-wise
        scheduar.step()

def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []

    with torch.no_grad():
        for bi, data  in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = data['ids']
            token_type_ids = data['token_type_ids']
            mask = data['mask']
            targets =  data['targets']
            
            #  Set everything to device later
            
            # extract the outputs
            outputs = model(
                    ids = ids,
                    mask = mask,
                    token_type_ids = token_type_ids
                    )

            fin_targets.extend(targets.cpu().detech().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detech().numpy().tolist())

    return fin_outputs fin_targets
            



