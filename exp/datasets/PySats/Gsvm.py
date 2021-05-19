from jnius import JavaClass, MetaJavaClass, JavaMethod, JavaMultipleMethod, cast, autoclass

SizeBasedUniqueRandomXOR = autoclass(
    'org.spectrumauctions.sats.core.bidlang.xor.SizeBasedUniqueRandomXOR')
JavaUtilRNGSupplier = autoclass(
    'org.spectrumauctions.sats.core.util.random.JavaUtilRNGSupplier')
Bundle = autoclass(
    'org.spectrumauctions.sats.core.model.Bundle')

class _Gsvm(JavaClass, metaclass=MetaJavaClass):
    __javaclass__ = 'org/spectrumauctions/sats/core/model/gsvm/GlobalSynergyValueModel'

    # TODO: I don't find a way to have the more direct accessors of the DefaultModel class. So for now, I'm mirroring the accessors 
    #createNewPopulation = JavaMultipleMethod([
    #    '()Ljava/util/List;',
    #    '(J)Ljava/util/List;'])
    setNumberOfNationalBidders = JavaMethod('(I)V')
    setNumberOfRegionalBidders = JavaMethod('(I)V')
    createWorld = JavaMethod(
        '(Lorg/spectrumauctions/sats/core/util/random/RNGSupplier;)Lorg/spectrumauctions/sats/core/model/gsvm/GSVMWorld;')
    createPopulation = JavaMethod(
        '(Lorg/spectrumauctions/sats/core/model/World;Lorg/spectrumauctions/sats/core/util/random/RNGSupplier;)Ljava/util/List;')

    population = {}
    goods = {}
    efficient_allocation = None

    def __init__(self, seed, number_of_national_bidders, number_of_regional_bidders):
        super().__init__()
        if seed:
            rng = JavaUtilRNGSupplier(seed)
        else:
            rng = JavaUtilRNGSupplier()

        self.setNumberOfNationalBidders(number_of_national_bidders)
        self.setNumberOfRegionalBidders(number_of_regional_bidders)
        
        world = self.createWorld(rng)
        self._bidder_list = self.createPopulation(world, rng)

        # Store bidders
        bidderator = self._bidder_list.iterator()
        while bidderator.hasNext():
            bidder = bidderator.next()
            self.population[bidder.getId()] = bidder
        
        # Store goods
        goods_iterator = self._bidder_list.iterator().next().getWorld().getLicenses().iterator()
        count = 0
        while goods_iterator.hasNext():
            good = goods_iterator.next()
            assert good.getId() == count
            count += 1
            self.goods[good.getId()] = good
        
    
    def get_bidder_ids(self):
        return self.population.keys()

    def get_good_ids(self):
        return self.goods.keys()

    def calculate_value(self, bidder_id, goods_vector):
        assert len(goods_vector) == len(self.goods.keys())
        bidder = self.population[bidder_id]
        bundle = Bundle()
        for i in range(len(goods_vector)):
            if goods_vector[i] == 1:
                bundle.add(self.goods[i])
        return bidder.calculateValue(bundle).doubleValue()
