package com.example.riskplatform;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@EnableScheduling
@SpringBootApplication
@MapperScan("com.example.riskplatform.mapper")
public class RiskPlatformApplication {
    public static void main(String[] args) {
        SpringApplication.run(RiskPlatformApplication.class, args);
    }
}
