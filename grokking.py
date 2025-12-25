import numpy as np
from tinygrad import Tensor, nn, TinyJit

from tqdm import tqdm

from transf import Transformer


def gen_data(cyclic_size: int) -> tuple[np.ndarray, np.ndarray]:
    # generate cyclic addition data points
    data = []
    out = []
    for i in range(cyclic_size - 1):
        for j in range(cyclic_size - 1):
            data.append(
                np.array(
                    [
                        i,
                        j,
                        cyclic_size
                        - 1,  # separator (must be < cyclic_size for one_hot)
                    ]
                )
            )

            out.append(np.array((i + j) % (cyclic_size - 1)))
    return np.array(data), np.array(out)


def main():
    Tensor.manual_seed(0)
    # define model
    # this includes the seperator symbol
    # so we learn a sym-1 modular arithmetic
    syms = 63

    # we only ever have x 'SEP' y
    maxlen = 3

    # params
    layers = 2
    embed_dim = 128
    num_heads = 4
    ff_dim = embed_dim * 4

    model = Transformer(syms, maxlen, layers, embed_dim, num_heads, ff_dim)

    # define data
    data, out = gen_data(syms)
    true = Tensor(out).one_hot(syms)
    data = Tensor(data)

    # shuffle then split into train and test
    split_ratio = 0.4
    order = Tensor.randperm(len(data))
    x = data[order]
    y = true[order]
    split = int(len(data) * split_ratio)

    train_x = x[:split]
    train_y = y[:split]
    test_x = x[split:]
    test_y = y[split:]

    opt = nn.optim.AdamW(nn.state.get_parameters(model), weight_decay=1)

    bs = 512

    @TinyJit
    def train_step():
        samples = Tensor.randint(bs, low=0, high=train_x.shape[0])
        X = train_x[samples]
        Y = train_y[samples]
        loss = model.forward(X)[:, -1].cross_entropy(Y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    for step in tqdm(range(5000)):
        with Tensor.train():
            train_step()

        if step % 100 == 0:
            # train loss
            pred = model.forward(train_x)[:, -1]
            loss = pred.cross_entropy(train_y)
            # test step
            test_pred = model.forward(test_x)[:, -1]
            test_loss = test_pred.cross_entropy(test_y)
            print(
                f"step: {step}, train loss: {loss.item():.2f}, test loss: {test_loss.item():.2f}"
            )

    # measure accuracy
    pred = model.forward(test_x)[:, -1]
    acc = pred.argmax(axis=1).eq(test_y.argmax(axis=1)).float().mean()
    print(f"final test accuracy: {acc.item():.2f}")


if __name__ == "__main__":
    main()
