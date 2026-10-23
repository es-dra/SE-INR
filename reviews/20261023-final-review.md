# Claude 改稿终审（2026-10-23）

审阅版本：323670ad950ab2689752ce13def38d47864ca7b7。结论：数字与版式通过，主张措辞需修改；不需新增实验。按分工未改正文。

## 结果与贡献

第 17 阶段描述性端点仍为 11/12，J6/J2=0.94–2.47；第 18 阶段原生相对常数/打乱为 +0.100/+0.097 dB，两控制均通过 ±0.03 dB 等价门槛，四项预测通过，独立复核记录完整。第二、三条贡献成立；新稿第一条和统领机制的措辞强于证据。重新编译成功，5 页，第 5 页仅参考文献；逐页视觉检查无重叠或裁切，19 个嵌入字体、无 Type 3。重编译 PDF 与提交 PDF 逐页文本一致。

## 必须修改的三处

1. **P1：Nyquist 边界与绝对化。** main.tex:9–17、29–31、147 把收益归为 above-Nyquist，并称 decoder form does not / only。Nyquist 是 u=1；04-training-prior/score_v4.json 的 LIIF ×2/3/4 在 [0.75,1) 已提高 0.0082/0.0108/0.0084，低频也有至多 0.0013 的变化。标题建议用 “High-Frequency Capacity and Scale-Paired Learning”；贡献建议 “Capacity gains concentrate near and above the LR Nyquist limit; tested decoder forms share recovery profiles.” 将 only 改 mainly，结论避免把已测设计的相近曲线写成新解码器不可能改变的容量上限。
2. **P1：训练收益不排除推理滤波。** main.tex:17、29 用 “rather than as an inference-time filter” 和 “for learning them” 推出机制。第 18 阶段区分正确配对与保留维度的控制，未排除原生 cell 的推理滤波，也未定位其收益为 above-Nyquist prior。建议统领句：“Correct scale pairing provides a training benefit, while equal-step probes reveal sustained inference-time responses.” 保留正面训练结论，删去排他机制推断；与第 4 节“不识别唯一内部机制”一致。
3. **P2：表 2 改写了通过的命题。** main.tex:117 的 “capacity raises R only above u≈1” 不是冻结门槛。04-training-prior/protocol_v4.json 的 R1 检验 [1,2) 提升与 [0,0.75) 差异 ≤0.01，未限制 [0.75,1)。建议行名 “training: gains at 1≤u<2; |ΔR|≤0.01 below u=0.75”，标明 12/12、9/9 为每个解码器的计数；若概括编码器也须分别对应阶段 5 的门槛。不得将新的绝对结论记作预注册通过。

## 风险与下一步

Claude 只需完成上述定点措辞修改，再编译确认 4+1 页；无需训练或追加分析。11/12 继续标为描述性，Set14 失败继续保留。Codex 已更新当前文件的验证哈希及 README 标题，PLAN 记录复核状态；科学终审与技术编译通过分开记录。
