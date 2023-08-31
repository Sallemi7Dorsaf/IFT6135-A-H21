import numpy as np
import torch
import torchvision


def one_hot(y, n_classes=10):
    return np.eye(n_classes)[y]


def load_cifar10(root, flatten=False):
    """
    Usage example:
    > train_data, valid_data, test_data = load_cifar10("/data", flatten=True)
    > train_x, train_y = train_data
    where both train_x and train_y are numpy arrays
    train_x.shape == (40000, 3072) or train_x.shape == (40000, 3, 32, 32)
    train_y.shape == (40000, 10), one-hot format
    :param root: path where the cifar10 dataset will be downloaded, e.g. "/tmp/data/"
    :param flatten: When True, dataset is reshaped to (num_examples, 3072), otherwise shape is (num_examples, 3, 32, 32)
    :return: train, valid and test set in numpy arrays
    """
    transform = torchvision.transforms.ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(root, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root, train=False, transform=transform, download=True)

    # randomly split train into train/valid
    perm = np.random.RandomState(seed=1).permutation(
        range(len(train_dataset)))  # fix seed to have same split every time.
    x = [train_dataset[i][0] for i in perm]  # train_dataset.data[perm]
    y = [one_hot(train_dataset[i][1]) for i in perm]
    train_x, train_y = x[:40000], y[:40000]
    valid_x, valid_y = x[40000:], y[40000:]
    test_x = [test_dataset[i][0] for i in range(len(test_dataset))]
    test_y = [one_hot(test_dataset[i][1]) for i in range(len(test_dataset))]

    # convert to numpy arrays after stacking
    train_x = torch.stack(train_x).cpu().numpy()
    train_y = np.stack(train_y)
    valid_x = torch.stack(valid_x).cpu().numpy()
    valid_y = np.stack(valid_y)
    test_x = torch.stack(test_x).cpu().numpy()
    test_y = np.stack(test_y)

    if flatten:
        train_x = train_x.reshape(-1, 32 * 32 * 3)
        valid_x = valid_x.reshape(-1, 32 * 32 * 3)
        test_x = test_x.reshape(-1, 32 * 32 * 3)

    # Package everything
    train_data = train_x, train_y
    valid_data = valid_x, valid_y
    test_data = test_x, test_y

    return train_data, valid_data, test_data


class NN(object):
    def __init__(self,
                 hidden_dims=(784, 256),
                 epsilon=1e-6,
                 lr=0.01,
                 batch_size=64,
                 seed=1,
                 activation="relu",
                 data=None
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if data is None:
            # for testing, do NOT remove or modify
            self.train, self.valid, self.test = (
                (np.random.rand(400, 3072), one_hot(np.random.randint(0, 10, 400))),
                (np.random.rand(400, 3072), one_hot(np.random.randint(0, 10, 400))),
                (np.random.rand(400, 3072), one_hot(np.random.randint(0, 10, 400)))
        )
        else:
            self.train, self.valid, self.test = data


    def initialize_weights(self, dims):        
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            n_in=all_dims[layer_n-1]
            n_out=all_dims[layer_n]
            low=-1*np.sqrt(6/(n_in+n_out))
            high=np.sqrt(6/(n_in+n_out))
            weight=np.random.uniform(low,high,size=(n_in,n_out))
            self.weights[f"W{layer_n}"] = weight
            
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def relu(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            x=np.maximum(x,0.0,dtype=np.float64)
            x[x>0]=1.0
            return x
            #pass
        # WRITE CODE HERE

        #pass
        return np.maximum(x,0.0,dtype=np.float64)

    def sigmoid(self, x, grad=False):
        x=np.array(x,np.float64)
        sig = lambda x : np.exp(x)/(1+np.exp(x))
        if grad:
            # WRITE CODE HERE
            return sig(x)*(1-sig(x)) 
        # WRITE CODE HERE
        
        return sig(x)

    def tanh(self, x, grad=False):
        x=np.array(x,np.float64)
        t=(np.exp(x)-np.exp(-1*x))/(np.exp(x)+np.exp(-1*x))
        if grad:
            # WRITE CODE HERE
          return 1-t**2
        # WRITE CODE HERE
        
        return t

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            # WRITE CODE HERE
            return self.relu(x,grad)
        elif self.activation_str == "sigmoid":
            # WRITE CODE HERE
            return self.sigmoid(x,grad)
        elif self.activation_str == "tanh":
            # WRITE CODE HERE
            return self.tanh(x,grad)
        else:
            raise Exception("invalid")
        return 0

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        # WRITE CODE HERE
        X=x
        def soft_max_one_element(x):
            e_x = np.exp(x - np.max(x)) 
            return e_x / e_x.sum()
        if len(x.shape)>1 :
            sof=[soft_max_one_element(x) for x in X]
            return np.array(sof)
        else :
            return soft_max_one_element(x)

    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionnary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        # WRITE CODE HERE
        print("shape z 0is {}: ".format(cache["Z0"].shape))
        for layer_n in range(1, self.n_hidden + 2):
          z=cache[f"Z{layer_n-1}"]
          w=self.weights[f"W{layer_n}"]
          b=self.weights[f"b{layer_n}"]
          print(f"shape W {layer_n} is : {w.shape}")
          print(f"shape b {layer_n} is : {b.shape}")

          a= np.dot(z,w)+b   #pre-activation
          z=self.activation(a,grad=False)
          print(f"shape activation Z  {layer_n} is : {z.shape}")
          print(f"shape pre-activation A {layer_n} is : {a.shape}")
          cache[f"Z{layer_n}"]=z 
          cache[f"A{layer_n}"]= a     

        cache[f"Z{layer_n}"]=self.softmax(cache[f"A{layer_n}"])
        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        # grads is a dictionnary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        # WRITE CODE HERE
        grads[f"dA{self.n_hidden+1}"]=output-labels
        print(output)
        print(labels)
        print(output-labels)
        print("output Z_{} {}".format(self.n_hidden+1, output.shape))

        print("grad A_{} {}".format(self.n_hidden+1, grads[f"dA{self.n_hidden+1}"].shape))

        


        for i,layer_n in enumerate(reversed(range(1, self.n_hidden+2 ))):
          #print(self.n_hidden+1-i)
          m=len(labels)
          grads[f"dW{layer_n}"]=(1/m)*np.dot(cache[f"Z{layer_n-1}"].T,grads[f"dA{layer_n}"])
          print("grad W_{} {}".format(self.n_hidden+1-i,grads[f"dW{layer_n}"].shape))
          grads[f"db{layer_n}"]=(1/m)*np.sum(grads[f"dA{layer_n}"],axis=0,keepdims=True)
          print("grad b_{} {}".format(self.n_hidden+1-i,grads[f"db{layer_n}"].shape))
          
          if (layer_n>1) :
            grads[f"dZ{layer_n-1}"]=np.dot(grads[f"dA{layer_n}"],self.weights[f"W{layer_n}"].T )
            print("grad Z_{} {}".format(self.n_hidden-i,grads[f"dZ{layer_n-1}"].shape))         
            grads[f"dA{layer_n-1}"]=np.multiply(grads[f"dZ{layer_n-1}"],self.activation(cache[f"A{layer_n-1}"],grad=True))
            print("grad A_{} {}".format(self.n_hidden-i,grads[f"dA{layer_n-1}"].shape))

        return grads
  

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            
            self.weights[f"W{layer}"]=self.weights[f"W{layer}"] - (self.lr)*grads[f"dW{layer}"]
            self.weights[f"b{layer}"]=self.weights[f"b{layer}"] - (self.lr)*grads[f"db{layer}"]
            


    # def one_hot(self, y, n_classes=None):
    #     n_classes = n_classes or self.n_classes
    #     return np.eye(n_classes)[y]

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        # WRITE CODE HERE
        prediction=np.array(prediction)
        labels=np.array(labels)
        N = prediction.shape[0]
        ce = -np.sum(labels*np.log(prediction))/N
        return ce

    def compute_loss_and_accuracy(self, X, y):
        one_y = y
        y = np.argmax(y, axis=1)  # Change y to integers
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = y_train
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                # WRITE CODE HERE
                cache=self.forward(minibatchX)
                grads=self.backward(cache, minibatchY)
                
                self.update(grads)

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy
