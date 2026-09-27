package com.example.riskplatform.service.rag;

import com.example.riskplatform.common.BusinessException;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.apache.poi.xwpf.usermodel.XWPFParagraph;
import org.apache.poi.xwpf.usermodel.XWPFTable;
import org.apache.poi.xwpf.usermodel.XWPFTableRow;
import org.apache.tika.Tika;
import org.apache.tika.metadata.Metadata;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Locale;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

@Service
public class KnowledgeFileService {
    private static final Set<String> ALLOWED = Set.of(
            "txt", "md", "csv", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "rtf");
    private static final Set<String> ZIP_CONTAINERS = Set.of("docx", "xlsx", "pptx");
    private final Path storageRoot;
    private final boolean ocrEnabled;
    private final String ocrCommand;
    private final int ocrMaxPages;
    private final Tika tika = new Tika();

    public KnowledgeFileService(
            @Value("${app.knowledge.storage-dir:./data/knowledge-files}") String storageDir,
            @Value("${app.knowledge.ocr-enabled:false}") boolean ocrEnabled,
            @Value("${app.knowledge.ocr-command:tesseract}") String ocrCommand,
            @Value("${app.knowledge.ocr-max-pages:10}") int ocrMaxPages) {
        this.storageRoot = Path.of(storageDir).toAbsolutePath().normalize();
        this.ocrEnabled = ocrEnabled;
        this.ocrCommand = ocrCommand;
        this.ocrMaxPages = Math.max(1, ocrMaxPages);
    }

    public StoredFile store(Long documentId, int version, MultipartFile file) {
        if (file == null || file.isEmpty()) throw new BusinessException("请选择非空文件");
        String original = safeName(file.getOriginalFilename());
        String ext = extension(original);
        if (!ALLOWED.contains(ext)) throw new BusinessException("不支持的文件类型: " + ext);
        try {
            Files.createDirectories(storageRoot);
            Path temp = Files.createTempFile(storageRoot, "upload-", ".tmp");
            try {
                file.transferTo(temp);
                validate(temp, ext);
                String sha = sha256(temp);
                String relative = documentId + "/v" + version + "/" + sha + "-" + original;
                Path target = resolve(relative);
                Files.createDirectories(target.getParent());
                Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
                return new StoredFile(relative.replace('\\', '/'), sha, Files.size(target), tika.detect(target), original);
            } finally {
                Files.deleteIfExists(temp);
            }
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            throw new BusinessException("文件保存失败: " + e.getMessage());
        }
    }

    public Extraction extract(String storageUri, String mimeType) throws Exception {
        Path file = resolve(storageUri);
        if (!Files.isRegularFile(file)) throw new IOException("原始文件不存在: " + storageUri);
        String ext = extension(file.getFileName().toString());
        String text;
        if ("docx".equals(ext)) text = extractDocx(file);
        else if ("xlsx".equals(ext) || "xls".equals(ext)) text = extractWorkbook(file);
        else try (InputStream in = Files.newInputStream(file)) { text = tika.parseToString(in, new Metadata(), -1); }
        String warning = null;
        if ("pdf".equals(ext) && normalize(text).length() < 80 && ocrEnabled) {
            String ocr = ocrPdf(file);
            if (!ocr.isBlank()) text = ocr;
            else warning = "PDF 文本较少，OCR 未提取到有效内容";
        } else if ("pdf".equals(ext) && normalize(text).length() < 80) {
            warning = "疑似扫描版 PDF；如需识别图片文字，请开启 OCR";
        }
        text = normalize(text);
        if (text.isBlank()) throw new IOException("未提取到可索引文本");
        double quality = quality(text);
        if (quality < 55 && warning == null) warning = "解析质量偏低，请检查原文件排版或编码";
        return new Extraction(text, mimeType, quality, warning);
    }

    public void deleteQuietly(String storageUri) {
        if (storageUri == null || storageUri.isBlank()) return;
        try { Files.deleteIfExists(resolve(storageUri)); } catch (Exception ignored) { }
    }

    private void validate(Path file, String ext) throws Exception {
        byte[] head = new byte[4];
        try (InputStream in = Files.newInputStream(file)) { in.read(head); }
        if (head[0] == 'M' && head[1] == 'Z') throw new BusinessException("拒绝可执行文件");
        if (ZIP_CONTAINERS.contains(ext)) validateZip(file);
    }

    private void validateZip(Path file) throws Exception {
        long total = 0;
        int entries = 0;
        try (ZipFile zip = new ZipFile(file.toFile())) {
            var it = zip.entries();
            while (it.hasMoreElements()) {
                ZipEntry e = it.nextElement();
                entries++;
                if (entries > 10000) throw new BusinessException("压缩容器条目过多");
                String n = e.getName().toLowerCase(Locale.ROOT);
                if (n.contains("../") || n.startsWith("/") || n.contains("vbaproject.bin"))
                    throw new BusinessException("文件包含危险路径或宏代码");
                long size = Math.max(0, e.getSize());
                long compressed = Math.max(1, e.getCompressedSize());
                total += size;
                if (total > 200L * 1024 * 1024 || (size > 10L * 1024 * 1024 && size / compressed > 100))
                    throw new BusinessException("压缩容器膨胀率异常");
            }
        }
    }

    private String extractDocx(Path path) throws Exception {
        StringBuilder out = new StringBuilder();
        try (InputStream in = Files.newInputStream(path); XWPFDocument doc = new XWPFDocument(in)) {
            for (XWPFParagraph p : doc.getParagraphs()) appendLine(out, p.getText());
            for (XWPFTable table : doc.getTables()) {
                for (XWPFTableRow row : table.getRows()) {
                    appendLine(out, row.getTableCells().stream().map(c -> normalize(c.getText()))
                            .reduce((a, b) -> a + " | " + b).orElse(""));
                }
            }
        }
        return out.toString();
    }

    private String extractWorkbook(Path path) throws Exception {
        StringBuilder out = new StringBuilder();
        DataFormatter formatter = new DataFormatter(Locale.CHINA);
        try (InputStream in = Files.newInputStream(path); Workbook wb = WorkbookFactory.create(in)) {
            for (Sheet sheet : wb) {
                appendLine(out, "工作表: " + sheet.getSheetName());
                for (Row row : sheet) {
                    StringBuilder line = new StringBuilder();
                    for (Cell cell : row) {
                        if (!line.isEmpty()) line.append(" | ");
                        line.append(formatter.formatCellValue(cell));
                    }
                    appendLine(out, line.toString());
                }
            }
        }
        return out.toString();
    }

    private String ocrPdf(Path path) {
        StringBuilder out = new StringBuilder();
        try (PDDocument doc = PDDocument.load(path.toFile())) {
            PDFRenderer renderer = new PDFRenderer(doc);
            int pages = Math.min(doc.getNumberOfPages(), ocrMaxPages);
            for (int i = 0; i < pages; i++) {
                Path image = Files.createTempFile("knowledge-ocr-", ".png");
                try {
                    ImageIO.write(renderer.renderImageWithDPI(i, 180), "png", image.toFile());
                    Process p = new ProcessBuilder(ocrCommand, image.toString(), "stdout", "-l", "chi_sim+eng")
                            .redirectErrorStream(true).start();
                    try (BufferedReader reader = new BufferedReader(
                            new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
                        reader.lines().forEach(line -> appendLine(out, line));
                    }
                    if (p.waitFor() != 0) return "";
                } finally { Files.deleteIfExists(image); }
            }
        } catch (Exception e) { return ""; }
        return out.toString();
    }

    private Path resolve(String relative) {
        Path p = storageRoot.resolve(relative).normalize();
        if (!p.startsWith(storageRoot)) throw new BusinessException("非法文件路径");
        return p;
    }

    private static String safeName(String name) {
        String n = name == null ? "document.txt" : Path.of(name).getFileName().toString();
        n = n.replaceAll("[^\\p{L}\\p{N}._() -]", "_");
        return n.length() > 180 ? n.substring(n.length() - 180) : n;
    }

    private static String extension(String name) {
        int i = name.lastIndexOf('.');
        return i < 0 ? "" : name.substring(i + 1).toLowerCase(Locale.ROOT);
    }

    private static String sha256(Path path) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        try (InputStream in = Files.newInputStream(path)) {
            byte[] buf = new byte[8192];
            for (int n; (n = in.read(buf)) > 0; ) md.update(buf, 0, n);
        }
        return HexFormat.of().formatHex(md.digest());
    }

    private static String normalize(String value) {
        if (value == null) return "";
        return value.replace("\u0000", "").replace("\r\n", "\n").replace('\r', '\n')
                .replaceAll("[\\t\\x0B\\f ]+", " ").replaceAll("\\n{3,}", "\n\n").trim();
    }

    private static double quality(String text) {
        int useful = 0;
        int replacement = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (Character.isLetterOrDigit(c) || (c >= 0x4e00 && c <= 0x9fff)) useful++;
            if (c == '\ufffd') replacement++;
        }
        double ratio = text.isEmpty() ? 0 : useful * 100.0 / text.length();
        double lengthScore = Math.min(30, Math.log10(Math.max(10, text.length())) * 8);
        return Math.max(0, Math.min(100, ratio + lengthScore - replacement * 2.0));
    }

    private static void appendLine(StringBuilder out, String line) {
        if (line != null && !line.isBlank()) out.append(line.trim()).append('\n');
    }

    public record StoredFile(String storageUri, String sha256, long size, String mimeType, String originalName) { }
    public record Extraction(String text, String mimeType, double qualityScore, String warning) { }
}
