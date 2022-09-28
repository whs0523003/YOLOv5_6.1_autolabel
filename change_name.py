import os

# 用于批量修改文件名

path = input('请输入文件夹路径(结尾加上/)：')

# 获取该目录下所有文件，存入列表中
filelist = os.listdir(path)

n = 346
n_recode = n

for i in filelist:
    # 设置旧文件名（就是路径+文件名）
    oldname = path + os.sep + filelist[n-n_recode]  # os.sep添加系统分隔符

    # 设置新文件名
    newname = path + os.sep + 'img' + str(n + 1) + '.jpg'

    os.rename (oldname, newname)  # 用os模块中的rename方法对文件改名
    print(oldname, '======>', newname)

    n += 1