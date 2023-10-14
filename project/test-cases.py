import numpy as np

def outer_it(x, y, out=None):
    mulop = np.multiply
    it = np.nditer([x, y, out], ['external_loop'],
            [['readonly'], ['readonly'], ['writeonly', 'allocate']],
            op_axes=[list(range(x.ndim)) + [-1] * y.ndim,
                     [-1] * x.ndim + list(range(y.ndim)),
                     None])
    with it:
        for (a, b, c) in it:
            mulop(a, b, out=c)
        return it.operands[2]

a = np.arange(2)+1
b = np.arange(3)+1
outer_it(a,b)



softmax_y = np.array([[              #3d array
                        [2.9, -5.6], 
                        [6.4, -8.9]
                    ],
                    [
                        [-5.2, 7.5],
                        [6.2, -3.1]
                    ],
                    [
                        [-6.5, 10.5],
                        [4.7, -3.8]
                    ]]
            )
dim_y = 2

softmax_z = np.arange(72) #4d array
softmax_z = np.reshape(softmax_z, (4,3,2,3))
dim_z = 4

print(f"\n Softmax Input: \n {softmax_z} \n")
res_softmax_z =  Softmax(torch.tensor(softmax_z), dim_z)
print(f"Custom Output: \n {res_softmax_z}")

torch_Softmax_z = torch.nn.Softmax(dim_z)
print("Pytorch Output: \n")
print(torch_Softmax_z(torch.tensor(softmax_z)))

