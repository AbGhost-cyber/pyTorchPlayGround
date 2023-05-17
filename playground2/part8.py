from datetime import datetime

n_chans1 = 32


# - `conv1` layer:
# - Number of parameters = (3 * 3 * 3 * `n_chans1`) + `n_chans1` = 84 * `n_chans1`
# - `conv2` layer:
# - Number of parameters = (`n_chans1` * 2 * 2 * (`n_chans1` // 2)) + (`n_chans1` // 2) = `n_chans1`^2 + (`n_chans1` // 2)
# - `fc1` layer:
# - Number of parameters = (8 * 8 * `n_chans1` // 2 * `n_chans2`) + `n_chans2` = 3,840 * `n_chans2` + `n_chans2`
# - `fc2` layer:
# - Number of parameters = `n_chans2` * 2 + 2 = 66

def training_loop_l2reg(n_epochs, optimizer, model, loss_fn,
                        train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum()
                          for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.now(), epoch,
                loss_train / len(train_loader)))


if __name__ == '__main__':
    print()
