-- 首次启动时建库并授权。
--
-- 只在数据目录为空时执行一次（Docker 官方镜像的 entrypoint 行为）。
-- 改了这个文件之后要 `docker compose down -v` 删掉数据卷才会重新执行。
--
-- 为什么建两个库：
--   codeinsight_v2       应用运行时用
--   codeinsight_v2_test  集成测试用，测试会 DELETE 全部表
--
-- 分开是 DEC-0041 的要求。集成测试读的是另一个环境变量
-- CODEINSIGHT_MYSQL_TEST_DATABASE，且库名必须含 "test"。这样「清空数据」
-- 这个动作在配置层面就无法指向应用库——比写注释提醒可靠得多。

CREATE DATABASE IF NOT EXISTS `codeinsight_v2`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;

CREATE DATABASE IF NOT EXISTS `codeinsight_v2_test`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;

-- MYSQL_USER 创建的账号默认只对 MYSQL_DATABASE 有权限，而我们没设那个变量，
-- 所以两个库的权限都要在这里显式授予。
--
-- 只给这两个库的权限，不给全局 —— 应用账号没有理由能读写别的库。
-- 主机部分用 '%' 是因为连接来自容器网络外（宿主机映射端口），
-- 从 MySQL 的角度看源地址不固定。
GRANT ALL PRIVILEGES ON `codeinsight_v2`.* TO 'codeinsight'@'%';
GRANT ALL PRIVILEGES ON `codeinsight_v2_test`.* TO 'codeinsight'@'%';

FLUSH PRIVILEGES;
