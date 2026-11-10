# Spring Boot 自动配置原理

## 一个注解背后的三件事

我们通常在启动类上写 `@SpringBootApplication`，它本身是一个组合注解，由三个注解构成：
`@SpringBootConfiguration`、`@EnableAutoConfiguration` 和 `@ComponentScan`。
其中 `@ComponentScan` 负责扫描当前包及其子包，把带有 `@Component`、`@Service`、`@Repository`、`@Controller`
这类注解的类注册成 Bean；真正让"自动配置"这个名字成立的是 `@EnableAutoConfiguration`。

## 自动配置类从哪里来

`@EnableAutoConfiguration` 内部通过 `@Import(AutoConfigurationImportSelector.class)` 生效。
这个选择器并不去扫描类路径上的所有类，而是读取类路径下各个 jar 包内的清单文件：

- Spring Boot 2.7 之前：读 `META-INF/spring.factories`，键是 `org.springframework.boot.autoconfigure.EnableAutoConfiguration`；
- Spring Boot 2.7 起：改为读 `META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports`，一行写一个全限定类名。

读出来的候选配置类有几百个，但绝大多数不会真正生效，因为每个配置类上都挂着条件注解。

## 条件装配：按需生效

常见的条件注解有以下几类：

- `@ConditionalOnClass` / `@ConditionalOnMissingClass`：判断类路径上是否存在某个类；
- `@ConditionalOnBean` / `@ConditionalOnMissingBean`：判断容器中是否已经存在某个 Bean；
- `@ConditionalOnProperty`：判断某个配置项是否等于指定值；
- `@ConditionalOnWebApplication`：判断当前是不是 Web 应用。

其中 `@ConditionalOnMissingBean` 是"约定优于配置"的关键：只要你自定义了一个同类型的 Bean，
自动配置就会主动退让，不会覆盖你的定义。这也是为什么引入 `spring-boot-starter-data-redis` 之后
什么都不用配就能拿到一个可用的 `RedisTemplate`，而一旦你自己定义了 `RedisTemplate`，
自动配置的那一个就自动失效。

## 想看看到底生效了哪些

启动时加 `--debug` 参数，或者在配置文件里打开
`logging.level.org.springframework.boot.autoconfigure=DEBUG`，
控制台会打印一份条件评估报告（Condition Evaluation Report），
分成 Positive matches 与 Negative matches 两部分，
可以直观看到某个自动配置类究竟是因为哪一个条件不满足而被跳过。

## 外部配置的优先级

同一个配置项可能出现在好几个地方，Spring Boot 按固定顺序决定谁说了算，
大致是：命令行参数 > 操作系统环境变量 > jar 包外部的 `application-{profile}.properties` >
jar 包内部的 `application-{profile}.properties` > 通用的 `application.properties`。
带 profile 的那份优先级高于通用那份，所以只要激活某个 profile，
就能用 `application-{profile}.properties` 覆盖掉通用配置里的同名项，而通用文件本身一个字都不用改。
