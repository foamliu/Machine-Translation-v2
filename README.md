# 英中机器文本翻译

评测英中文本机器翻译的能力。机器翻译语言方向为英文到中文。


## 依赖

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 数据集

我们使用AI Challenger 2017中的英中机器文本翻译数据集，超过1000万的英中对照的句子对作为数据集合。其中，训练集合占据绝大部分，验证集合8000对，测试集A 8000条，测试集B 8000条。

可以从这里下载：[英中翻译数据集](https://challenger.ai/datasets/translation)

![image](https://github.com/foamliu/Machine-Translation/raw/master/images/dataset.png)

## 用法

### 数据预处理
提取训练和验证样本：
```bash
$ python pre-process.py
```

### 训练
```bash
$ python train.py
```

要想可视化训练过程，在终端中运行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo
下载 [预训练模型](https://github.com/foamliu/Scene-Classification/releases/download/v1.0/model.85-0.7657.hdf5) 放在 models 目录然后执行:

```bash
$ python demo.py
```

<pre>
> we have a house to find  .
= 有房子要找。
< 我们要找房子了。
> i have   but i do n t  .
= 喝过，但是不喝。
< 我有但不行。
> yeah   i suppose you have  .
= 是啊，我猜是吧。
< 我想我是你的。
> i have a girl friend  .
= 我有女朋友了。
< 我有一个女朋友的。
> another rule  ? that  s rule number one  .
= 又是一条规则？这是规则一。
< 另一个规则？是的规则。
> are you asleep  ? yeah  .
= 你睡着了？是的。
< 你睡着了？嗯。
> i  m driving  . no way  .
= 我来开车。不行。
< 我开车。。不。
> just choose  !
= 你做个选择吧！
< 一定要选择！
> i have suspected him for ages  .
= 我早就怀疑他了。
< 我早就怀疑他了了。
> i  m a partner here  .
= 我是这儿的合伙人。
< 我在这里一个。。

> any news  ? nothing  .
= 有进展吗？没。
< 有什么消息吗？没什么。
> yeah   i bet you want a lawyer  .
= 好吧，我打赌你要律师。
< 你想律师律师律师律师。
> i  m going to stop the missile  .
= 我要去阻止导弹发射。
< 我去准备了导弹。
> we agreed to have dinner together  .
= 我们说好一起吃饭的。
< 我们说好一起吃饭的。
> was her husband
= 全是她丈夫
< 是她的丈夫
> do n t worry about dinner   all right  ?
= 别担心晚餐，好么？
< 不用担心担心的的吧吧？
> i  m dying  . i ca n t breathe  .
= 我要死了，无法呼吸。
< 我死了。不能呼吸。
> i have a new line   too
= 我也有个新电话，
< 我也有新的
> yeah   i  ll take a beer  .
= 好，给我来一支。
< 是的我我。。
> we can do this  .
= 我们就可以离开这儿了。
< 我们可以做。

> oh   is he okay  ?
= 哦，他还好吗？
< 他没事吧？
> i  m sorry i called you an idiot  .
= 对不起叫你们白痴。
< 我很抱歉我叫你个白痴。
> i  m helping you   don  .
= 我是在帮你。
< 我帮你你你。
> are you gon na complain the whole way  ?
= 你想一路抱怨过去吗？
< 你就会抱怨我？
> we did some research on you  .
= 我们调查了一下你的资料。
< 我们已经调查了了调查。
> yeah  ! that  s perfect  !
= 好啊！这个主意好！
< 是！！！！
> as many as i want  .
= 想吃多少就吃多少。
< 我想多。
> oh   i  m sorry   buddy  .
= 噢，我很抱歉，兄弟。
< 噢，我很抱歉。
> we have the greatest army in the world  .
= 我们有世界上最厉害的军队。
< 我们拥有世界世界上的部队。
> kid   you ai n t    .
= 小子，你不是21岁。
< 孩子你你。。


> i  m leaving that with you
= 我把它留给你
< 我和你一起
> i  m gon na sit right here  .
= 我就坐这儿不动。
< 我就坐在这里。
> we may be able to bring the memories
= 我们能把这种记忆
< 我们可能能把记忆记忆
> laid down some cash  .
= 给了这些钱。
< 把钱拿。。
> we found it   you guys  !
= 伙计们，咱们找到了！
< 我们找到了你们！
> we haven   t gotten any physical evidence  . . .
= 我们还没有任何的证据...
< 我们还没有证据证据证据证据。
> i have to get up too  .
= 我也要起床了。
< 我也得上去。
> at home   alone   no   wait  .
= 在家，一个人，等一下。
< 家里一个人不等。
> oh   god  . it  s mom  .
= 天啊，是妈妈。
< 哦，妈妈。妈妈。
> i  m on hold again  .
= 我拿着电话呢。
< 我又来了。

> i  m also her hero
= 我还是她的英雄，
< 我也她她英雄
> i  m the greatest  !
= 我们是伟大的！
< 我是最棒的！
> i  m gon na kill him  !
= 我想杀掉他！
< 我想杀了他！
> oh   daniel  . . . charming  .
= 噢，丹尼尔…漂亮。
< 丹尼尔。。。。。
> i  m not out of this relationship  .
= 我不会退出这个复杂关系的。
< 我不脱离这段关系。
> we have more important business in hand  .
= 我们还有更重要的事。
< 我们的事我们更重要的生意。
> just that no one can know about it  .
= 只是不能被人发现罢了。
< 只是没有人能知道这件事。
> we have to finish reading this statement
= 我们必须完成阅读这个声明
< 我们得得读一下读
> we have a very very busy day tomorrow  .
= 我们明天会很忙很忙。
< 明天我们有忙忙。
> are you sure you  re not coming  ?
= 你们真的不一起来？
< 你真的不来吗？

> i  m sorry  .
= 抱歉。
< 我很抱歉。
> we got a little held up  .
= 我们有点事情。
< 我们有点有点了。
> we need to put a tail on him  .
= 我们应该跟踪他。
< 我们需要跟踪他。
> the same to put him down
= 像这样放下，
< 把他的的
> i  m getting her prints anyway  .
= 可我就是要她的指纹。
< 我我检查她她的指纹。
> we just want to run some tests  .
= 我们只是想做些检测。
< 我们只是想检查检查。
> just admit it   just this once  .
= 承认吧就这一次。
< 只是承认这次一次。
> oh no  ! the clock  !
= 哦，不！时间到了！
< 不！闹钟！
> of course  . that  s a  . . .
= 当然去过啊。那里...
< 当然。。。。。
> exception   i get it  .
= 你抗议，我知道了。
< 我我明白。。

> anyway   it  s not like you  . . .
= 另外不管是你...
< 我说不像你。
> any way you cut the sandwich
= 反正你得给我问出来，
< 你不会把三明治三明治
> just wait a few more days  .
= 再多等几天吧。
< 再等几天。
> are you satisfied   you asshole  ?
= 你满意了吧？你这个混蛋？
< 你满意了你吗？
> are   are we  . . . in danger  ?
= 我们有危险吗？
< 我们的。。。危险？？
> we are short on time tonight  .
= 但是我们节目时间有限。
< 我们我们是短短。。
> we are going back to india right now  .
= 我们现在就回印度。
< 我们现在回到印度。。
> girls and boys were dragged off
= 女孩和男孩们被抓去
< 女孩们的孩子们
> yeah   i  ve thought that before  .
= 嗯，我过去想过这个。
< 是的我我的。。
> just go  . we  re losing time  .
= 快走。我们没时间了。
< 快走。我们损失了时间。

> we gave her too much freedom  !
= 我们给她太多自由了！
< 我们给了她太多了自由！
> i have the bullets  .
= 子弹在我手上。
< 我有子弹。
> we closed the deal today
= 今天他们接受了
< 今天我们不了了
> watch me  .
= 你看我写不写。
< 小心看着我。
> definitely  . . . hey  .
= 当然了... 嘿。
< 一定。。。嘿。
> apparently  . we can go now  .
= 显然。我们可以走了。
< 很好。我们现在走。。
> so i heard you married the mayor  .
= 听说你嫁给了市长。
< 我听说你你了市长。
> l call on you in the name of liberty
= 以自由的名义我呼吁你们
< 在叫我名义名义
> we need more people like you  .
= 我们很需要你这种人。
< 我们需要更像你的人。
> we might as well get comfortable  .
= 我们还是保持现状的好。
< 我们还是得舒服舒服。

> i  m dying   tom  .
= 我要死了汤姆。
< 我是在汤姆。
> we fight and we die  . . .
= 战斗死亡...
< 我们战斗然后我们死。
> are you listening to me  ? yeah  .
= 你在听我说吗？在。
< 听我说吗？是的。
> just give it to him   dad  .
= 给他就好了，爸。
< 他爸爸爸爸。。
> was he so bad even at home  !
= 他在家里是不是也这样坏的！
< 他甚至在在坏
> oh you do n t believe that  .
= 你自己也不信吧。
< 你不会相信的。
> oh   god   sorry  . sorry  .
= 哦，天，对不起对不起。
< 哦，对不起。对不起。
> anyway   have a good night  . night  .
= 晚安啦。晚安。
< 昨晚晚上了。。。
> are you two having fun  ?
= 你俩玩得好吗？
< 你们俩玩的开心吗？
> oh   fuck  ! god damn it  .
= 干！妈的！
< 噢！他妈的。

> of course  . where am i going  ?
= 当然了。去哪里接？
< 当然。。我要去哪？
> i  m not a father
= 我不是一名父亲
< 我不是一个父亲
> we can win  !
= 我们会成功的！
< 我们赢了！
> so cool down and sit down  .
= 冷静一下，坐下来。
< 冷静冷静坐下。
> so i might not be jewish  .
= 所以我有可能不是犹太人。
< 所以我不是犹太人。
> any other family  ?
= 家里还有其他人吗？
< 其他的家人？
> i have a deal to propose  .
= 我想提议你做个交易。
< 我有协议要好。
> i hear there was a certain transfer of power
= 我听说有某种权力的转移
< 我听说有某种能量
> apart from my mother   obviously
= 当然，除了我母亲外，
< 我母亲显然是
> i  m saying i love you  .
= 我是说，我爱你们。
< 我是说我爱你。

> know what happened here tonight  .
= 不会有人知道今晚的事。
< 今晚知道发生了什么。
> are you kidding me right now  ?
= 你在跟我开玩笑吗？
< 你在开玩笑吧？
> i have no plans to die today  .
= 我今天可没打算去死。
< 我今天没有计划。
> just lost his best friend
= 失去了最好的朋友，
< 他失去了了最好的朋友
> any luck searching the house  ?
= 房间里有什么发现吗？
< 有什么进展吗？
> so i  m going to go  .
= 我走了。
< 所以我要去。
> any reaction   a direction of some sort  .
= 反应、或指导方向之类。
< 任何什么都反应。
> of what he was afraid of
= 使他感到害怕
< 他害怕的
> king suites
= 特大床套房：
< 特大套房
> i  m sure that you would
= 我知道你肯定不会愿意，
< 我确信你会

> are you listening  ? just keep breathing  .
= 你听到了吗？保持呼吸。
< 听着吗？呼吸呼吸。。
> oh christ  . we  re dead  .
= 哦天啊，我们死定了。
< 天啊。我们了了。
> i  m coming as fast as i can  .
= 我马上来了。
< 我很快就能尽快。
> just more patient   it wo n t take long
= 在耐心点，不会太长时间的
< 会更耐心的更耐心
> so i said     i will give you
= 所以我说" 我用我的狗
< 所以我说我会给你
> we could have helped each other  .
= 我们就应该帮助对方。
< 我们可以可以帮助帮助。
> are you  . . . are n t you coming up  ?
= 你…… 你还不出来？
< 你在。。吗？？
> we let you go  .
= 我们放你们走。
< 我们就放你走。
> so do i   at home  .
= 我家里也是。
< 我在家就。。
> john was a good guy  .
= 约翰是个好人。
< 约翰约翰是个好男人。

</pre>