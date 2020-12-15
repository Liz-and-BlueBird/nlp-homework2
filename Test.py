from past.builtins import raw_input

from TextCNN import model_load, get_label


def get_input_label():
    """
    根据用户输入返回类别
    """
    # 载入模型
    state_dic = model_load('./model/256-50-[30, 30, 30]-[2, 3, 4]-10-0.01.pt')

    # 读取用户输入
    user_input = raw_input('请输入:')
    # 输入非空
    while user_input != '':
        # 获取类别
        label = get_label(state_dic['net'], state_dic['vocab'], state_dic['category'], user_input)
        # 打印
        print(label)
        # 继续监听输入
        user_input = raw_input('请输入:')


if __name__ == '__main__':
    get_input_label()
