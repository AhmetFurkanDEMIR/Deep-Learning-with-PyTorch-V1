import torch

import torch.nn as nn

import torch.optim as optim

import torch.utils.data

import torch.nn.functional as F

import torchvision

from torchvision import transforms

from PIL import Image

import imutils

import cv2

# veri setindeki resimler okunabilecek düzeydeyse,
# yani hatalı değilse dahil ediyoruz. 
def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False

# verilerin konumları
train_data_path = "./train/"
val_data_path = "./val/"
test_data_path = "./test/"

# resmleri vektör haline getirmek için işlemler.
img_transforms = transforms.Compose([
    
    # resimleri yeniden boyutlandırıyoruz
    transforms.Resize((64,64)),
    
    # resimleri tensör haline getiriyoruz    
    transforms.ToTensor(),
    
    # resimleri normalize ediyoruz. 0,1 aralığına
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225] )
    ])


# oluşturduğumuz transform ile resimleri yüklüyoruz, eğitim verisi
train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_transforms, is_valid_file=check_image)

# doğrulama verisi
val_data = torchvision.datasets.ImageFolder(root=val_data_path,transform=img_transforms, is_valid_file=check_image)

# test verisi
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=img_transforms, is_valid_file=check_image) 

# bir döngüde alınabilecek resim sayısı
batch_size=64

# verileri döngüye yani eğitime aktarmak için parçalıyoruz
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

val_data_loader  = torch.utils.data.DataLoader(val_data, batch_size=batch_size) 

test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size) 

# modelimiz
class SimpleNet(nn.Module):
    
    # katmanlar, girdi ve çıktı boyutları
    def __init__(self):
        
        # Tamamen bağlı katmanlar
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50,2)
        
        # ikili sıflandırma olduğu için çıktı boyutu iki
        # ya balık yada kedi
    
    # katmanlarda kullanılan aktivasyon fonksiyonları
    def forward(self, x):
        x = x.view(-1, 12288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# modeli tanımladık
simplenet = SimpleNet()


# eniyileme
# ağımızın girdisi olan veri ile oluşturduğu kaybı göz önünde bulundurarak,
# kendisini güncelleme mekanizması
optimizer = optim.Adam(simplenet.parameters(), lr=0.001)


# Pytorch Gpu versiyonu ve Cdu kuruluysa eğitim Gpu üzerinden gerçekleşir
# kurulu değilse Cpu
if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")

simplenet.to(device)


# modeli eğitme
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs, device):
    
    # eğitimin döngüs sayısı
    for epoch in range(epochs):
        
        training_loss = 0.0 # eğitim kaybı
        valid_loss = 0.0 # doğrulama kaybı
        
        # modelin eğitime başladığını kodumuza söylüyoruz.
        # eğitim verisi için geçerlidir.
        model.train()
        
        for batch in train_loader:
            
            # eniyileme
            optimizer.zero_grad()
            
            # model eğitim yığını kadar resmi alıp eğitime başlar.
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            # bir resmin kaybı, hatası hesaplanması
            
            loss = loss_fn(output, targets)
            loss.backward()
            
            # eniyileme, kaybı sıfıra yaklaştırmaya çalışmak
            optimizer.step()
            
            # eğitim kayıplarının toplamı
            training_loss += loss.data.item() * inputs.size(0)
        
        # bir döngüdeki ortalama kayıb 
        training_loss /= len(train_loader.dataset)
        
        # test veya doğrulam verisetinde eğitimi bildiriyoruz
        # model.train(mode = False) # bu şekilde de yapabilirsiniz. 
        model.eval()
        
        
        num_correct = 0 
        num_examples = 0
        
        for batch in val_loader:
            
            # model eğitim yığını kadar resmi alıp eğitime başlar.
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)

            # resmin kaybı
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
            
        # bir döngüdeki ortalama kayıb 
        valid_loss /= len(val_loader.dataset)
        
        # eğitimdeki döngü ve kayıplar hakkında bilgilendirmeler.
        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))
 
#eğitim fonksiyonunu çağırdık
# aldığı parametreler sırasıyla:
# model, eniyileme fonksiyonu, kayıp fonksiyonu, eğitim verisi, doğrulama verisi, döngü sayısı, eğitimin gereçkleşeceği ortam (Cpu,Gpu)
train(simplenet, optimizer,torch.nn.CrossEntropyLoss(), train_data_loader,val_data_loader, epochs=10, device=device)

# modeli test etmek için kaydettik.
torch.save(simplenet.state_dict(), "simplenet")  

#%% Modeli test etmek

# labeller
labels = ['cat','fish']

dataa = ["./test/fish/1.jpg","./test/fish/3.jpg","./test/cat/3.jpg","./test/cat/1.jpg"]

data = dataa[1]
    
# resmi çağırdık
img = Image.open(data)

# resmi işleme 
img = img_transforms(img)

image = cv2.imread(data)
orig = image.copy()

# ikili sınıflandırma yaptığımız için aktivasyon fonksiyonunu softmax olarak aldık
# ya balık yada kedi
prediction = F.softmax(simplenet(img),dim=1)
prediction = prediction.argmax()
print(labels[prediction]) 

if prediction >= 0.5:

	label = "fish"
	a = 100 * prediction

else:

	label = "cat"
	a = (1-prediction) * 100


label = "{}: {:.2f}%".format(label, a)
# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
 
