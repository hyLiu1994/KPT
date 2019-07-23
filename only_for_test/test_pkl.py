import pickle

f=open('./data/hdu/hdu_userLC_[15, 1000000, 0.06, 1]_problemLC_[10, 1000000, 0.02, 1]_timeLC_[2018-11-22 20.45.48, 2018-11-29 11.22.08]_OnlyRight_True_problemName2problemId.pkl','rb')
file=pickle.load(f)

print(file)