import scipy as sp

x = sp.array([[-11, 21, 31],
     [12,22,32],
      [-13,23,33]])
y = sp.array([1,-1,-1])

print x
print y
print ''
x = sp.concatenate((sp.ones((1,x.shape[1])), x))
w = sp.ones((x.shape[0]))/x.shape[0]

print x
print w

wrong = (sp.sign(w.dot(x)) != y).nonzero()[0]

print wrong

output = w.T.dot(x)
print output
print y

correct = sp.select(condlist=[output >0],choicelist=[output])
print correct
correct = output * y
print correct

