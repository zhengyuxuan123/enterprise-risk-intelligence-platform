package com.example.riskplatform.controller;

import com.example.riskplatform.common.ApiResponse;
import com.example.riskplatform.entity.KnowledgeDocument;
import com.example.riskplatform.entity.RagIngestJob;
import com.example.riskplatform.service.KnowledgeService;
import com.example.riskplatform.service.RagService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.util.List;

@RestController @RequestMapping("/api/knowledge") @RequiredArgsConstructor
public class KnowledgeController {
    private final KnowledgeService service;
    private final RagService rag;

    @GetMapping @PreAuthorize("hasAuthority('knowledge:read')")
    public ApiResponse<List<KnowledgeDocument>> list(@RequestParam(required=false) Long companyId) {
        List<KnowledgeDocument> rows=service.list(companyId);
        rows.forEach(d->d.setContent(d.getContent()!=null&&d.getContent().length()>500?d.getContent().substring(0,500)+"...":d.getContent()));
        return ApiResponse.ok(rows);
    }
    @GetMapping("/jobs") @PreAuthorize("hasAuthority('knowledge:read')")
    public ApiResponse<List<RagIngestJob>> jobs(@RequestParam(required=false) Long documentId){return ApiResponse.ok(service.jobs(documentId));}
    @PostMapping("/upload") @PreAuthorize("hasAuthority('knowledge:write')")
    public ApiResponse<KnowledgeDocument> upload(@RequestParam(required=false) Long companyId,@RequestParam(required=false) Long deptId,
        @RequestParam(required=false) String docType,@RequestParam(required=false) Integer securityLevel,
        @RequestParam(required=false) String title,@RequestPart("file") MultipartFile file){
        return ApiResponse.ok(service.upload(companyId,deptId,docType,securityLevel,title,file));
    }
    @PostMapping("/{id}/replace") @PreAuthorize("hasAuthority('knowledge:write')")
    public ApiResponse<KnowledgeDocument> replace(@PathVariable Long id,@RequestPart("file") MultipartFile file){return ApiResponse.ok(service.replace(id,file));}
    @PostMapping("/{id}/retry") @PreAuthorize("hasAuthority('knowledge:write')")
    public ApiResponse<Void> retry(@PathVariable Long id){service.retry(id);return ApiResponse.ok();}
    @GetMapping("/search") @PreAuthorize("hasAuthority('knowledge:read')")
    public ApiResponse<List<RagService.RagHit>> search(@RequestParam Long companyId,@RequestParam String q,@RequestParam(defaultValue="5") int topK){return ApiResponse.ok(rag.search(companyId,q,topK));}
    @DeleteMapping("/{id}") @PreAuthorize("hasAuthority('knowledge:write')")
    public ApiResponse<Void> delete(@PathVariable Long id){service.delete(id);return ApiResponse.ok();}
}
