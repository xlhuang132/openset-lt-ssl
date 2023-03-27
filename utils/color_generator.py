import colorsys
import random

def generate_ood_color():
    ood_color= color(tuple([0,0,0]))
    return  ood_color

def generate_colors(num,return_deep_group=False,return_light_group=False,alpha=0.5):
    colors=ncolors(num) 
    dark_colors,light_colors=[],[]
    if return_deep_group:# 在原基础上 产生一组同色系深色
        black=[0,0,0]
        dark_colors=[[int(alpha*color[i]+(1-alpha)*black[i]) for i in range(len(color))] for color in colors]
        colors.extend(dark_colors)
    if return_light_group:# 在原基础上 产生一组同色系浅色
        white=[255,255,255]
        light_colors=[[int(alpha*color[i]+(1-alpha)*white[i]) for i in range(len(color))] for color in colors]
        colors.extend(light_colors)
    colors = list(map(lambda x: color(tuple(x)),  colors)) 
    return colors
     
def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors

def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


# zhuanhuanwei 16 jinzhi
def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)


