# BOOTSTRAP.md — Agent 会话引导协议

> 用法：每次新 session 开头发一句「读 BOOTSTRAP.md 并按规则工作」即可。
> 本文件同时规定了「开局要恢复上下文」和「收尾要固化上下文」，让思维链不依赖对话记忆。

---

## 🟢 开局仪式（每个 session 开始时，自动执行，先做完再动手）

1. **拉取仓库现状**
   ```bash
   git status
   git log -10 --oneline
   git branch --show-current
   ```
2. **读取续接文件**
   - `cat STATUS.md`（若不存在，按下方模板创建骨架）
3. **恢复开发环境**（关键！session 切换后 bash 状态不保留）
   - 检查 GitHub 登录：`gh auth status` 或 `git ls-remote origin`
     - 失败则：用 SSH key（`~/.ssh` 会持久化），或重新 `gh auth login`
   - 依赖恢复：执行 `./setup.sh` 或 `pnpm install`
4. **向用户汇报（3–5 行）**：当前分支 / 上次进度 / 本会话目标 / 任何阻塞

---

## 🔵 工作约定（每次都遵守）

- **决策要落盘**：任何「为什么这么做」的理由写进 `STATUS.md` 的「决策记录」，不要只留在对话里
- **小步提交**：每完成一个里程碑就 `git commit`，message 写清意图（这就是可恢复的"思考链"）
- **分支隔离**：不同任务用 `feature/xxx`、`fix/yyy`，切 session 直接 `git checkout` 回到现场
- **代码即事实**：以仓库 + 工作区文件为唯一事实来源，不靠对话记忆

---

## 🔴 收尾仪式（每次结束前，自动执行）

1. **更新 `STATUS.md`**：已完成 / 进行中 / 下一步 / 踩坑 / 关键文件路径
2. **提交并推送**
   ```bash
   git add -A
   git commit -m "wip: <一句话说明本次进展>"
   git push
   ```
3. **若要切 session**：先完成上面两步再走，确保下一个 session 能无缝接上

---

## 📄 STATUS.md 模板（首次创建时用）

```markdown
# STATUS.md
## 当前目标
-

## 已完成
- [ ]

## 进行中
-

## 下一步（TODO）
1.

## 决策记录 / 踩坑
-

## 关键文件路径
-
```
