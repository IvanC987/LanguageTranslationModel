import torch
from Train import initiate_model

device = "cuda" if torch.cuda.is_available() else "cpu"
state = torch.load("TL_0.810-VL_0.955_state_dict.pth", map_location=torch.device(device))
model = initiate_model()[0].to(device)
model.load_state_dict(state)

while True:
    sentence = input("Enter src: ")
    print(model.translate(sentence, k=5))
    print("-----------------------------")
