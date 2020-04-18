import torch

print(torch.cuda.is_available()) # False çıkarsa CPU
# True çıkarsa GPU

print(torch.rand(2,2))
# basit bir tensör oluşturma.

tensor = torch.tensor([[5,6],[7,8]])
# tensör oluşturma

zeros_tensors = torch.zeros(5,5)
# sıfırlardan oluşan tensör

ones_tensor = torch.ones(2,2)
# birlerden oluşan tensör

ones2_tensors = torch.ones(2,2)

ones3_tensors = ones2_tensors+ones_tensor
# iki tensör toplama

x = torch.rand(1)

print(x.item()) # tensör içindeki sayıyı alır.

print(tensor)

print(zeros_tensors)

print(ones_tensor)

print(ones3_tensors)

cpu_tensor = torch.rand(2)

print(cpu_tensor.device) # işlemi CPU da yapıyor

"""

 -! işlemi Gpu da yapmaya zorluyoruz.
 -! bende pytorch cpu kurulu olduğu için hata verdi. 

gpu_tensor = cpu_tensor.to("cuda")

print(cpu_tensor.device)


"""

x = torch.rand(5,5)

print(x.max().item()) # tensördeki en büyük değer

# tensör boyutları

dört_b_tensor = torch.tensor([[5,9],[6,1],[8,0],[9,1]])

uc_b_tensorr = torch.tensor([[5,6,8],[4,6,7],[4,2,3]])

iki_b_tensorr = torch.tensor([[4,5,6,3],[5,6,9,8]])

bir_b_tensorr = torch.tensor([5,6])

,
# tensör tipi, (longtensor)
print(dört_b_tensor.type())

# tensör tipi, (floattensor)
float_tensor = torch.tensor([[5.,6.],[8.,1.]])

print(float_tensor.type())

# tensörün logaritmasını alır.
print(float_tensor.log())

tensor = torch.rand(784)

# (1) tensörün boyutunu yeniden şekillendirir
tensor = tensor.view(1,28,28)

# tensör boyututnu yazdırır
print(tensor.shape)

# (2) tensörün boyutunu yeniden şekillendirir
tensor = tensor.reshape(28,28,1)

print(tensor.shape)

tensora = torch.rand(640,480,3)

# tensörü yeniden şekillendirir.
tensora = tensora.permute(2,1,0)

print(tensora.shape)



