import six
import sys
sys.modules['sklearn.externals.six'] = six
import numpy as np
from id3 import Id3Estimator, export_text

nasdaq = np.genfromtxt('Nasdaq1YDaily.csv', delimiter=',')

features = ["Date", "Open", "High", "Low", "Close", "Adj", "Close", "Volume"]

y = np.array(["(3k,3k)"])

outlook = Id3Estimator()
outlook.fit(nasdaq, y, check_input=True)

print(export_text(outlook.tree_))
