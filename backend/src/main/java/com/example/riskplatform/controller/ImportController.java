package com.example.riskplatform.controller;

import com.example.riskplatform.common.ApiResponse;
import com.example.riskplatform.entity.ImportTask;
import com.example.riskplatform.service.ImportService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.util.List;

@RestController
@RequestMapping("/api/import")
@RequiredArgsConstructor
public class ImportController {
    private final ImportService service;

    @PostMapping("/excel")
    @PreAuthorize("hasAuthority('import:write')")
    public ApiResponse<ImportTask> excel(@RequestParam String importType, @RequestPart("file") MultipartFile file) {
        return ApiResponse.ok(service.importExcel(importType, file));
    }

    @GetMapping("/history")
    @PreAuthorize("hasAuthority('import:write')")
    public ApiResponse<List<ImportTask>> history() { return ApiResponse.ok(service.history()); }
}
