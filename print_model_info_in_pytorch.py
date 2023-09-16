import torch


class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()

        self.lnput = torch.nn.Linear(3, 1024, 1024)
        self.Flatten = torch.nn.Flatten()
        self.output = torch.nn.Linear(10500 + 8000 + (512 * 512))
        self.activation = torch.nn.ReLU()

        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.lnput(x)

        x = self.Flatten(x)
        x = self.output(x)

        return x


tinymodel = TinyModel()

print("The model:")
print(tinymodel)

print("\n\nJust one layer:")
print(tinymodel.linear2)

print("\n\nModel params:")
for param in tinymodel.parameters():
    print(param)

print("\n\nLayer params:")
for param in tinymodel.linear2.parameters():
    print(param)
