package com.example.riskplatform.entity;
import com.baomidou.mybatisplus.annotation.*;import lombok.Data;import java.time.*;
/**
 * RAG 语料块入库状态（{@code rag_chunk_state}）。
 *
 * <p>「写入即索引」的记账表：记录每个语料块的内容指纹与入库状态。
 * 它不存正文——正文在 Lucene 索引里（{@code RagIndexStore}），这里只存足以做增量判定的元信息。</p>
 *
 * <p>三个不可替代的用途：
 * <ol>
 *   <li><b>增量判定</b>：hash 没变就不重新向量化（remote 模式下这就是省钱闸门）。</li>
 *   <li><b>删除定位</b>：业务表是物理删除，删完就查不到了，只能靠这里的 chunk_id 反查去索引里剔除。</li>
 *   <li><b>失败补偿与巡检</b>：FAILED 的重跑；state 有、业务表没有的 → 剔除。</li>
 * </ol>
 */
@Data @TableName("rag_chunk_state") public class RagChunkState{
    @TableId(type=IdType.AUTO) private Long id;
    private String chunkId;
    private String sourceType;
    private String sourceId;
    private Long companyId;
    private String contentHash;
    private String status;
    private Integer retryCount;
    private String error;
    private LocalDateTime updatedAt;

    public static final String ST_PENDING = "PENDING";
    public static final String ST_INDEXED = "INDEXED";
    public static final String ST_FAILED = "FAILED";
}
