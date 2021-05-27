import javax.json.Json;
import javax.json.JsonObjectBuilder;

public class NamedEntity {
    private final String name;
    private final String entityType;
    private final int offset0;
    private final int offset1;

    public NamedEntity(String name, String entityType,  int offset0, int offset1) {
        this.name = name;
        this.entityType = entityType;
        this.offset0 = offset0;
        this.offset1 = offset1;
    }

    public String getName() {
        return name;
    }

    public String getEntityType() {
        return entityType;
    }

    public int getOffset0() {
        return offset0;
    }

    public int getOffset1() {
        return offset1;
    }

    @Override
    public String toString() {
        return "NamedEntity{" +
                "name='" + name + '\'' +
                ", entityType='" + entityType + '\'' +
                '}';
    }

    public JsonObjectBuilder toJsonObject() {
        JsonObjectBuilder result = Json.createObjectBuilder();
        result.add("ner", name).add("type", entityType);
        return result;
    }
}
